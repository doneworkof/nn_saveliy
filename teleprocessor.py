import telebot as tb
from telebot import types
from lightlang import Data, UserData
import lightlang as ll
from main import *
import random
from io import BytesIO
import json
import re
from image_search import img_search
from textblob import TextBlob

token = '5252227212:AAEqHc8GE5fRNclV78NJBPxN5guzN4XqWWc'
acceptable_content_types = ['text', 'sticker', 'photo', 'audio', 'video']

with open('scriptlib.json', 'r', encoding='utf8') as f:
    lib = json.load(f)


class scriptfunc:
    def __init__(self, content_types, first):
        self.first = first
        self.content_types = content_types.split(',')
        self.parent = None
    
    def past_init(self, func):
        self.func = func
        return self

    def setup_parent(self, parent):
        self.parent = parent

    def analisys(self, msg):
        return msg.content_type in self.content_types

    def __call__(self, data):
        if not self.parent:
            raise Exception('ScriptFunc must have a parent')
        return self.func(self.parent, data)

def ScriptFunction(content_types, first=False):
    scrfunc = scriptfunc(content_types, first)
    return scrfunc.past_init

class RequestsProcessor:
    def __init__(self, processing_func):
        self.requests = []
        self.busy = False
        self.processing_func = processing_func

    def process(self):
        self.busy = True
        while self.requests:
            req = self.requests.pop(0)
            self.processing_func(req)
        self.busy = False

    def append_req(self, req):
        self.requests.append(req)
        if not self.busy:
            self.process()

class TelegramBot:
    def __init__(self, beh):
        self.beh = beh
        self.core = tb.TeleBot(token)
        self.session = {}
        self.requests = RequestsProcessor(self.process_msg)

        self.new_msg = self.core.message_handler(
            content_types=acceptable_content_types
        )(self.new_msg)

    def type_error(self, data):
        print('type error')

    def on_new_user(self, userdata):
        for unit in self.beh:
            unit.on_new_userdata(userdata)

    def process_msg(self, msg):
        if msg.chat.id not in self.session:
            username = msg.from_user.username
            new_user = UserData(username)
            self.on_new_user(new_user)
            self.session[msg.chat.id] = new_user
        
        user = self.session[msg.chat.id]

        data = Data(
            user=user,
            msg=msg,
            core=self.core,
            script_fired=False,
            bot=bot,
            history=[],
            answer=[],
            flags={}
        )

        runthrough(self.beh, data)
   
    def new_msg(self, msg):
        print('New request created!')
        self.requests.append_req(msg)

    def run(self):
        self.core.polling(non_stop=True)

def construct(*options):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    items = [
        types.KeyboardButton(option) for option in options
    ]
    markup.add(*items)
    return markup

class ScriptUnit:
    def __init__(self, **params):
        self.params = params
        self.first_func = self.get_first_func()
        if not self.first_func:
            raise Exception(
                'There is no first script function in ScriptUnit'
            )
        for func in self.get_all_funcs():
            func.setup_parent(self)
        self.start()

    def start(self):
        pass

    def on_new_user(self, user):
        pass

    def analysis(self, data):
        return True

    def run(self, data):
        return self.first_func(data)

    def get_first_func(self):
        for func in self.get_all_funcs():
            if func.first:
                return func
        return None

    def get_all_funcs(self):
        result = []
        for key in dir(self):
            item = getattr(self, key)
            if type(item) == scriptfunc:
                result.append(item)
        return result

    def _analysis(self, data):
        if data.msg.content_type not in self.first_func.content_types:
            return False
        return self.analysis(data)

    def past_analysis(self, data):
        return True

class DefaultDialogProcessor(ScriptUnit):
    def past_analysis(self, data):
        return len(data.ready_units) == 1

    def on_new_user(self, user):
        bot.init_userdata(user)

    @ScriptFunction('text', first=True)
    def main(self, data):
        try:
            answer = bot.answer(
                data.msg.text, data.user
            )
        except:
            return
        
        data.core.send_message(data.msg.chat.id, answer)

class MemesProcessor(ScriptUnit):
    def generate_random_query(self):    
        rnd_keyword = random.choice(self.keywords)
        rus = en_to_rus.translate(f'memes about {rnd_keyword}')
        return rus
        
    def get_memes(self):
        print('Collecting memes...')
        self.memes = []
        count = random.randint(3, 5)
        for i in range(count):
            print(f'Collecting {i + 1} batch of memes...')
            query = self.generate_random_query()
            imgs = img_search(query, 5)
            self.memes += imgs
        print('Collected memes successfully.')
    
    def update_cycle(self):
        self.tick -= 1
        if self.tick <= 0:
            self.tick = 30
            self.get_memes()

    def start(self):
        with open('memes_keywords.bf', 'rb') as f:
            self.keywords = pickle.load(f)
        self.tick = 0
        self.update_cycle()

    def analysis(self, data):
        content = data.msg.text
        return re.search('(скинь|сбрось|хочу).*мем', content)

    @ScriptFunction('text', first=True)
    def main(self, data):
        rnd_img = random.choice(self.memes)
        bio = BytesIO()
        bio.name = 'meme.jpeg'
        rnd_img.save(bio, 'JPEG')
        bio.seek(0)
        data.core.send_photo(data.msg.chat.id, bio)
        self.update_cycle()

class TextPreprocessor(BehUnit):
    def analysis(self, data):
        return data.msg.content_type == 'text'
    
    def apply(self, data):
        text = data.msg.text
        translation = rus_to_en.translate(text)
        
        data.append(
            original_user_input=text,
            user_input=ll.simplify(data.msg.text),
            input_mood=TextBlob(translation).sentiment.polarity,
            translated=translation
        )

class ScriptCore(BehUnit):
    def start(self):
        if 'units' not in self.params:
            raise 'Script core must have units'
    
    def on_new_userdata(self, userdata):
        userdata['dominant_script'] = None
    
    def on_type_error(self, data):
        pass

    def apply(self, data):
        user = data.user

        if user['dominant_script'] is not None:
            if data.msg.content_type not in user['dominant_script'].content_types:
                self.on_type_error()
            else:
                user['dominant_script'] = user['dominant_script'](data)
                data.script_fired = True
            return
        
        ready_units = [
            unit for unit in self.params['units']
            if unit._analysis(data)
        ]

        ready_units = [
            unit for unit in ready_units
            if unit.past_analysis(data)
        ]

        if not ready_units:
            return
        
        random_unit = random.choice(ready_units)
        user['dominant_script'] = random_unit.run(data)
        data.script_fired = True

class StandardProcessor(BehUnit):
    def analysis(self, data):
        return not data.script_fired or conclude(50)
    
    def on_new_userdata(self, userdata):
        bot.init_userdata(userdata)
    
    def apply(self, data):
        bot.passthrough(data)

class ChunksConnector(BehUnit):
    def apply(self, data):
        if len(data.answer) == 1:
            chunk = data.answer[0]
            data.answer = chunk.normalize()
            return
        chunks = sorted(data.answer, key=lambda x: x.priority, reverse=True)
        if conclude(0.4):
            chunks = swap(chunks, times=1)
        answer = ''
        for i, chunk in enumerate(chunks):
            sep = '' if i + 1 == len(chunks) else ' '
            answer += chunk.normalize() + sep
        data.answer = capitalize(answer)

class Executor(BehUnit):
    def analysis(self, data):
        return data.answer
    
    def apply(self, data):
        data.core.send_message(data.msg.chat.id, data.answer)

tbot = TelegramBot([
    TextPreprocessor(),
    ScriptCore(units=[
        MemesProcessor()
    ]),
    StandardProcessor(),
    ChunksConnector(),
    SmileAttacher(),
    Executor()
])

tbot.run()


