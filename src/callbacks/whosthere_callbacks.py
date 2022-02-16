import os

from whos_there.callback import NotificationCallback
from whos_there.senders.telegram import TelegramSender


class TelegramNotification(NotificationCallback):
    def __init__(self):
        super().__init__(
            senders=[TelegramSender(chat_id=os.getenv("TG_CHAT_ID"), token=os.getenv("TG_TOKEN"))]
        )
