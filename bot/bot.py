import asyncio
import io
import logging
import os

import aiohttp
import cv2
import numpy as np
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spotter.bot")

BOT_TOKEN = os.getenv("BOT_TOKEN")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")

if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN is not set. Copy .env.example to .env and add your token."
    )

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


class AppStates(StatesGroup):
    main_menu = State()
    waiting_for_analysis_geo = State()
    waiting_for_analysis_photo = State()
    waiting_for_training_photo = State()


def get_main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔍 Анализ локации")],
            [KeyboardButton(text="📁 Загрузить примеры для обучения")],
            [KeyboardButton(text="⚙️ Запустить переобучение ИИ")],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def get_cancel_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="❌ Отмена")]],
        resize_keyboard=True,
    )


async def _download_photo(message: types.Message) -> bytes:
    photo_file = await bot.get_file(message.photo[-1].file_id)
    buffer = io.BytesIO()
    await bot.download_file(photo_file.file_path, buffer)
    return buffer.getvalue()


@dp.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "🧠 Добро пожаловать в панель управления **Spotter AI**.\n\n"
        "Выберите необходимое действие в меню ниже:",
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown",
    )


@dp.message(F.text == "❌ Отмена")
async def cmd_cancel(message: types.Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Действие отменено. Возврат в главное меню.",
        reply_markup=get_main_keyboard(),
    )


# --- РЕЖИМ 1: АНАЛИЗ ЛОКАЦИИ ---


@dp.message(F.text == "🔍 Анализ локации")
async def start_analysis(message: types.Message, state: FSMContext) -> None:
    await state.set_state(AppStates.waiting_for_analysis_geo)
    geo_keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(
                    text="📍 Отправить текущую геопозицию",
                    request_location=True,
                )
            ],
            [KeyboardButton(text="❌ Отмена")],
        ],
        resize_keyboard=True,
    )
    await message.answer(
        "Шаг 1: Нажмите на кнопку ниже, чтобы передать текущие координаты GPS.",
        reply_markup=geo_keyboard,
    )


@dp.message(AppStates.waiting_for_analysis_geo, F.location)
async def process_analysis_geo(message: types.Message, state: FSMContext) -> None:
    lat = message.location.latitude
    lon = message.location.longitude
    acc = message.location.horizontal_accuracy or 5.0

    await state.update_data(latitude=lat, longitude=lon, accuracy=acc)
    await state.set_state(AppStates.waiting_for_analysis_photo)

    await message.answer(
        f"📍 Координаты сохранены: {lat:.5f}, {lon:.5f}\n\n"
        "Шаг 2: Теперь отправьте **фотографию ракурса местности** "
        "для поиска ориентиров.",
        reply_markup=get_cancel_keyboard(),
        parse_mode="Markdown",
    )


@dp.message(AppStates.waiting_for_analysis_photo, F.photo)
async def process_analysis_photo(
    message: types.Message, state: FSMContext
) -> None:
    user_data = await state.get_data()
    await message.answer("🔄 Запрос отправлен на сервер. ИИ анализирует изображение...")

    photo_bytes = await _download_photo(message)

    async with aiohttp.ClientSession() as session:
        url = f"{BACKEND_BASE_URL}/api/v1/analyze"
        form = aiohttp.FormData()
        form.add_field("latitude", str(user_data["latitude"]))
        form.add_field("longitude", str(user_data["longitude"]))
        form.add_field("accuracy", str(user_data["accuracy"]))
        form.add_field(
            "photo", photo_bytes, filename="scan.jpg", content_type="image/jpeg"
        )

        try:
            async with session.post(url, data=form) as response:
                if response.status != 200:
                    await message.answer(
                        "❌ Ошибка сервера аналитики.",
                        reply_markup=get_main_keyboard(),
                    )
                    await state.clear()
                    return
                result = await response.json()
        except Exception as exc:
            await message.answer(
                f"❌ Нет связи с сервером: {exc}",
                reply_markup=get_main_keyboard(),
            )
            await state.clear()
            return

    visual_landmarks = result.get("visual_landmarks", [])
    geo_pred = result.get("geo_prediction", {})

    nparr = np.frombuffer(photo_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    for landmark in visual_landmarks:
        box = landmark["box_pixels"]
        conf = landmark["confidence"]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
        cv2.putText(
            img,
            f"Target: {int(conf * 100)}%",
            (box[0], max(box[1] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    _, img_encoded = cv2.imencode(".jpg", img)
    processed_photo_bytes = img_encoded.tobytes()

    response_text = (
        "📋 **Анализ завершен:**\n\n"
        f"🧭 **Маршрут:** {geo_pred.get('action_required')}\n"
    )
    if geo_pred.get("predicted_lat"):
        response_text += (
            "🎯 **Ожидаемая точка:** "
            f"`{geo_pred['predicted_lat']:.5f}, {geo_pred['predicted_lon']:.5f}`"
        )

    input_file = BufferedInputFile(processed_photo_bytes, filename="result.jpg")
    await message.answer_photo(
        photo=input_file,
        caption=response_text,
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown",
    )
    await state.clear()


# --- РЕЖИМ 2: ЗАГРУЗКА ПРИМЕРОВ ДЛЯ ОБУЧЕНИЯ ---


@dp.message(F.text == "📁 Загрузить примеры для обучения")
async def start_training_upload(
    message: types.Message, state: FSMContext
) -> None:
    await state.set_state(AppStates.waiting_for_training_photo)
    await message.answer(
        "📥 Режим сбора датасета.\n\n"
        "Отправьте мне фотографию-пример (желательно с четким ориентиром "
        "по центру). Файлы сохранятся в базу обучения на сервере.\n\n"
        "После окончания загрузки всех фото нажмите кнопку '❌ Отмена' "
        "для возврата.",
        reply_markup=get_cancel_keyboard(),
    )


@dp.message(AppStates.waiting_for_training_photo, F.photo)
async def process_training_photo(
    message: types.Message, state: FSMContext
) -> None:
    photo_bytes = await _download_photo(message)

    async with aiohttp.ClientSession() as session:
        url = f"{BACKEND_BASE_URL}/api/v1/upload_example"
        form = aiohttp.FormData()
        form.add_field(
            "photo",
            photo_bytes,
            filename=f"train_{message.photo[-1].file_id[:8]}.jpg",
            content_type="image/jpeg",
        )

        try:
            async with session.post(url, data=form) as response:
                if response.status == 200:
                    await message.answer(
                        "✅ Фото успешно сохранено в базу данных ИИ. "
                        "Можете отправить следующее."
                    )
                else:
                    await message.answer(
                        "⚠️ Сервер принял файл, но возникла ошибка "
                        "при сохранении."
                    )
        except Exception as exc:
            await message.answer(f"❌ Ошибка отправки на бэкенд: {exc}")


# --- ТРИГГЕР ПЕРЕОБУЧЕНИЯ ---


@dp.message(F.text == "⚙️ Запустить переобучение ИИ")
async def trigger_ai_training(message: types.Message) -> None:
    await message.answer(
        "🔄 Отправлен запрос на запуск переобучения нейросети. "
        "Это может занять некоторое время..."
    )

    async with aiohttp.ClientSession() as session:
        url = f"{BACKEND_BASE_URL}/api/v1/train"
        try:
            async with session.post(url) as response:
                res_data = await response.json()
                if response.status == 200:
                    await message.answer(
                        "🚀 Процесс самообучения запущен на сервере в фоновом "
                        "режиме! Нейросеть обновляет веса под ваши паттерны."
                    )
                else:
                    detail = res_data.get("detail", "Неизвестный сбой")
                    await message.answer(f"❌ Ошибка: {detail}")
        except Exception as exc:
            await message.answer(
                f"❌ Не удалось связаться с сервером обучения: {exc}"
            )


async def main() -> None:
    logger.info("Starting Spotter AI bot, backend=%s", BACKEND_BASE_URL)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
