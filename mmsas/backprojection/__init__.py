import requests

def log_to_webhook(message):
    webhook = 'https://discord.com/api/webhooks/1217693168234008606/VxbqGnx5jeo7WVO-futht66b3WNHYV25ifJNI8Slkky8ZmPvIVqzLQCg2VKCTzH0cFi0'
    requests.post(webhook, {
        'content': message
    })