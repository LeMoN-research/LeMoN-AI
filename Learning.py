from ADsnetwork import LeMoN_AI
from time import time
from prefetch_generator import BackgroundGenerator
from telebot import TeleBot
from utils import get_data, iterate_minibatches
import config

bot = TeleBot(config.telegram_token)
def send_log(message):
    bot.send_message(config.channel, message)

send_log("Создаем модель...")
model = LeMoN_AI()

send_log("Грузим данные...")
points, music = get_data("ok_dat.npy")

send_log("""Настраиваем параметры:
epoch = 100
batch_size = 1000
time_leght = 20
""")
epoch = 100
batch_size = 1000
time_leght = 20

iterate_mini_set = {
    "points": points,
    "music": music,
    "batch_size": batch_size,
    "block_size": time_leght
}

message = """Эпоха: {}
Время: {} H
Средняя ошибка(square error): {}"""

send_log("Тренируемся")
try:
    for epoch in range(1, epoch+1):
        n = 0
        loss = 0
        st = time()
        for music_, shift, stp, delta_mov in BackgroundGenerator(iterate_minibatches(**iterate_mini_set)):
            loss += model.train(music_.reshape((-1, time_leght, 200)), shift.reshape((-1, 19, 38*3)), stp, delta_mov.reshape((-1, 38*3)))
            n += 1
        t = time() - st
        send_log(message.format(epoch, t, loss/(n*1.)))
        send_log("Сохраняем веса")
        try:
            model.save()
        except Exception as e:
            send_log("Error when save: {}".format(e))
except Exception as e:
    send_log("Error: {}".format(e))

send_log("!!!Готово!!!")