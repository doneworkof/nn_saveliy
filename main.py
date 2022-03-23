print('IMPORTING MODULES...')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from lightlang import *
from tensorflow.keras.models import load_model
import numpy as np
from random import random, choice, randint
import datetime
from dirty_slang_detect import dirty_slang_test
from deep_translator import GoogleTranslator
from textblob import TextBlob
from string import punctuation
import pickle
import re
from threading import Thread
from language import *

print('PREPARING TRANSLATORS...')

rus_to_en = GoogleTranslator(source='ru', target='en')
en_to_rus = GoogleTranslator(source='en', target='ru')

print('LOADING LANGUAGE UTILS...')

lp = LanguageProcessor()

debug = True

class InputMoodProcessor(BehUnit):
    def analysis(self, data):
        return abs(data.input_mood) >= 0.3

    def apply(self, data):
        mood = data.input_mood
        factor = 0
        mrk = ''
        line = ''

        if -0.5 < mood <= -0.3:
            factor = 3
            mrk = random_punct_mark('weak')
            line = random_line('sad')
        elif mood <= -0.5:
            factor = 5
            mrk = random_punct_mark('strong')
            line = random_line('agressive')
        elif 0.3 <= mood < 0.5:
            factor = 3
            mrk = random_punct_mark('plain')
            line = random_line('thankful')
        elif mood >= 0.5:
            factor = 5
            mrk = random_punct_mark(
                choice(['plain', 'strong'])
            )
            line = random_line('glad')
        else:
            return

        if line[-1] == '?':
            line, mrk = mark_filter(line)

        data.answer.append(
            Chunk(line, mark=mrk, priority=0.75, mood=mood * factor)
        )

class ChunksMoodCounter(BehUnit):
    def apply(self, data):
        data.mood = sum([chunk.mood for chunk in data.answer])

class InnerMoodProcessor(BehUnit):
    def on_new_userdata(self, userdata):
        userdata['mood'] = 0
        userdata['patience'] = 10
    
    def analysis(self, data):
        return not Parting in data.history
        
    def apply(self, data):
        data.bot.apply_mood_to(data.mood, data.user)

        if data.user['mood'] <= -30 or data.user['patience'] <= 0:
            main = choice(lib['parting']['main'])
            addition = choice(lib['lines']['agressive'])
            result = combine(main, addition, submark=',', swap_prob=0.3)
            mrk = random_punct_mark('strong')
            data.answer.append(
                Chunk(result, mark=mrk, priority=0.8, mood=-1)
            )
            data.bot.disconnect_from(data.user)

        elif data.user['patience'] < 10 and conclude(0.5):
            data.bot.apply_patience_to(1, data.user)

class DirtySlangCatcher(BehUnit):
    def analysis(self, data):
        return dirty_slang_test(data.user_input)

    def apply(self, data):
        main = choice(lib['lines']['dirty'])
        data.answer.append(
            Chunk(main, priority=0.3, mood=-1.5)
        )

class Protector(BehUnit):
    def analysis(self, data):
        return len(data.answer) < 1

    def apply(self, data):
        data.bot.apply_patience_to(-1, data.user)
        data.answer += [
            Chunk(choice(lib['lines']['empty']), mood=-2)
        ]

class InsultAttacher(BehUnit):
    def analysis(self, data):
        return data.input_mood <= -0.4
    
    def apply(self, data):
        if conclude(0.5):
            sample = choice(lib['insult']['samples'])
            word = choice(lib['insult']['words'])
            phrase = sample.replace('*', word)
        else:
            phrase = choice(lib['insult']['phrases'])
        

        posmark = random_punct_mark(None)
        qmark = random_punct_mark(None, question=True)

        phrase, mark = question_filter(phrase, posmark, qmark=qmark)

        data.answer.append(
            Chunk(phrase, mark=mark, mood=-2.5, priority=0.75)
        )

class Greeting(BehUnit):
    def on_new_userdata(self, userdata):
        userdata['greeted'] = False

    def analysis(self, data):
        if Parting in data.history:
            return False
        for g in lib['greeting']['main']:
            if g in data.user_input:
                return True
        return False

    def apply(self, data):
        if data.user['greeted']:
            self.repeating(data)
            return

        main = choice(lib['greeting']['main'])

        if main == 'добрый*':
            time = datetime.datetime.now().hour
            if time >= 23 or time <= 4:
                main = 'Доброй ночи'
            elif time >= 5 and time <= 11:
                main = 'Доброе утро'
            elif time >= 12 and time <= 17:
                main = 'Добрый день'
            else:
                main = 'Добрый вечер'

        chunks = [Chunk(main, priority=1)]

        if conclude(0.5):
            question = choice(lib['greeting']['questions'])
            question, _ = mark_filter(question)
            mrk = random_punct_mark(None, question=True)
            if conclude(0.6):
                chunks[0].insert(question, new_mark=mrk)
            else:    
                chunks.append(
                    Chunk(question, mark=mrk)
                )


        data.answer += chunks
        data.user['greeted'] = True

    def repeating(self, data):
        data.answer.append(
            Chunk(choice(lib['greeting']['repeating']), mood=-3)
        )
        data.bot.apply_patience_to(-1, data.user)

class SmileAttacher(BehUnit):
    def apply(self, data):
        if data.mood >= 0.3:
            category = 'happy'
        elif -0.1 <= data.mood < 0:
            category = 'disappointed'
        elif -0.3 <= data.mood < -0.1:
            category = 'sad'
        elif data.mood < -0.3:
            category = 'angry'
        else:
            return

        data.answer += ' ' + choice(lib['smiles'][category])

class Parting(BehUnit):
    def analysis(self, data):
        for p in lib['parting']['main']:
            if p in data.user_input:
                if p == 'пока' and data.user_input != 'пока' and 'bye' not in data.translated:
                    continue
                return True
        return False
                
    def apply(self, data):
        answer = choice(lib['parting']['main'])
        if conclude(0.55):
            addition = choice(lib['parting']['additions'])
            answer = combine(answer, addition)
            
        data.answer.append(
            Chunk(answer, priority=0.6)
        )
        data.bot.disconnect_from(data.user)

class SpamHandler(BehUnit):
    def on_new_userdata(self, userdata):
        userdata['history'] = []

    def analysis(self, data):
        for h in data.user['history']:
            if string_similarity(data.user_input, h) >= 0.8:
                return True
        return False

    def update(self, data):
        histobj = data.user['history']
        histobj.append(data.user_input)
        if len(histobj) > 5:
            histobj.pop(0)

    def on_reject(self, data):
        self.update(data)
    
    def apply(self, data):
        data.flags['repeating'] = True
        self.update(data)

class LaughAttacher(BehUnit):
    def apply(self, data):
        chunk = choice(data.answer)
        main = generate_laugh()
        chunk.insert(main, position=0)

class Neural(BehUnit):
    def start(self):
        self.core_name = ''
        self.durability = 0
        self.refresh_core()

    def excercise(self):
        self.core.predict(
            np.zeros((1, seq_len, vector_len))
        )

    def refresh_core(self):
        print('REFRESHING...')
        paths = os.listdir('cores/')
        while True:
            new_core_name = choice(paths)
            if new_core_name != self.core_name:
                self.core_name = new_core_name
                break
        self.core = load_model('cores/' + self.core_name)
        self.durability = randint(15, 20)
        self.excercise()
    
    def refresh_core_async(self):
        newthr = Thread(target=self.refresh_core)
        newthr.start()
    
    def apply(self, data):
        mat = np.array(lp.line_to_matrix(data.translated)).reshape(1, seq_len, -1)
        output = self.core.predict(mat).reshape(seq_len, -1)
        indices = np.apply_along_axis(np.argmax, 1, output)
        raw_words = lp.wordlist_frame.index[indices].tolist()
        words = []
        is_question = False
        for raw_w in raw_words:
            if raw_w == '<end>':
                break
            elif raw_w == '<qend>':
                is_question = True
                break
            elif raw_w == '<empty>':
                continue
            elif words and words[-1] == raw_w:
                continue
            else:
                words.append(raw_w)
        if len(words) == 0:
            return
        generated = ' '.join(words) + ('?' if is_question else '')
        translated = en_to_rus.translate(generated)
        if is_question:
            translated = translated.replace('?', '')
            mrk = random_punct_mark(None, question=True)
        else:
            mrk = random_punct_mark(None, question=False)
        data.answer.append(
            Chunk(translated, mark=mrk, priority=0.6)
        )
        self.durability -= 1
        if self.durability <= 0:
            self.refresh_core_async()

    def analysis(self, data):
        return self.durability > 0 and not flag(data.flags, 'repeating') \
               and (len(data.answer) == 0 or conclude(0.55))

class Alternative(BehUnit):
    def start(self):
        self.max_msg_len = 6
        self.forget_prob = 0.05
        self.remember_prob = 0.65
        try:
            self.load()
        except:
            self.words_relation = {}
        self.tick = 15

    def load(self):
        with open('alternative.bf', 'rb') as f:
            self.words_relation = pickle.load(f)

    def save(self):
        with open('alternative.bf', 'wb') as f:
            pickle.dump(self.words_relation, f)

    def analysis(self, data):
        return Neural not in data.history or conclude(0.2)

    def preprocess(self, data):
        extracted, _ = lp.extract(data.translated, with_punct=True)
        return list(filter(
            lambda x: x[0] not in ['?', '!', '.'], extracted
        ))

    def remember_new_rel(self, seq):
        for i, word in enumerate(seq[:-1]):
            if not conclude(self.remember_prob):
                continue
            elif word not in self.words_relation:
                self.words_relation[word] = []
            self.words_relation[word].append(
                seq[i + 1]
            )

    def forget_old_rel(self):
        keys_to_delete = []
        for r in self.words_relation:
            idx_to_del = [
                i for i in range(
                    len(self.words_relation[r])
                ) if conclude(self.forget_prob)
            ]
            for idx in idx_to_del[::-1]:
                self.words_relation[r].pop(idx)
            if len(self.words_relation[r]) == 0:
                keys_to_delete.append(r)

        for ktd in keys_to_delete:
            del self.words_relation[ktd]

    def _random_startword(self):
        filt = list(filter(
            lambda x: x[0] not in punctuation, self.words_relation.keys()
        ))
        if len(filt) == 0:
            return
        return choice(filt)

    def generate(self):
        if len(self.words_relation) == 0:
            return
        mxlen = min(self.max_msg_len, len(self.words_relation))
        if mxlen < 2:
            return
        length = randint(2, mxlen)
        last = self._random_startword()
        if last is None:
            return
        generated = last
        for i in range(length):
            if last not in self.words_relation:
                break
            next = choice(self.words_relation[last])
            if next[0] in punctuation:
                generated += next
            else:
                generated += ' ' + next
            last = next
        return mark_filter(generated)

    def update_saving_cycle(self):
        self.tick -= 1
        if self.tick <= 0:
            self.tick = 15
            self.save()

    def apply(self, data):
        extracted = self.preprocess(data)
        self.forget_old_rel()
        self.remember_new_rel(extracted)
        output = self.generate()
        if output is None:
            return
        generated, mark = output
        if not mark or mark[0] not in ['.', '?', '!']:
            mark = random_punct_mark(
                None, question=conclude(0.35)
            )
        generated = en_to_rus.translate(generated)
        data.answer.append(
            Chunk(generated, mark=mark)
        )
        self.update_saving_cycle()

class StateQP(BehUnit):
    def on_new_userdata(self, userdata):
        userdata['stateqp_happened'] = False
    def analysis(self, data):
        return triggers_check(data.translated, lib['state_qp']['triggers'])
    def repeating(self, data):
        data.flags['repeating'] = True
    def apply(self, data):
        if data.user['stateqp_happened']:
            self.repeating(data)
            return
        if conclude(0.4):
            # frozen phrases method
            answer = choice(lib['state_qp']['frozen'])
            answer, moodsymb = answer[1:], answer[0]
            factor = {
                '-': -1,
                '=': 0,
                '+': 1
            }[moodsymb]
        else:
            # layouts method
            layout = choice(lib['state_qp']['layouts'])
            factor = randint(-1, 1)
            rating = random_rating(factor)
            answer = layout.replace('*', rating)

        data.answer.append(
            Chunk(
                answer,
                mark=random_punct_mark(None),
                mood=1.5 * factor,
                priority=0.85
            )
        )
        data.user['stateqp_happened'] = True

class NameQP(BehUnit):
    def on_new_userdata(self, userdata):
        userdata['nameqp_happened'] = False
    def analysis(self, data):
        return triggers_check(data.translated, lib['name_qp']['triggers'])
    def repeating(self, data):
        data.flags['repeating'] = True
    def apply(self, data):
        if data.user['nameqp_happened']:
            self.repeating()
            return
        layout = choice(lib['name_qp']['layouts'])
        result = layout.replace('*', data.bot.name)
        data.answer.append(
            Chunk(result, mark=random_punct_mark(None), priority=0.75)
        )
        data.user['nameqp_happened'] = True

class DoingQP(BehUnit):
    def on_new_userdata(self, userdata):
        userdata['doingqp_happened'] = False
    def analysis(self, data):
        return triggers_check(data.translated, lib['doing_qp']['triggers'])
    def apply(self, data):
        if data.user['doingqp_happened']:
            data.flags['repeating'] = True
            return
        rnd = choice(lib['doing_qp']['answers'])
        data.answer.append(
            Chunk(rnd, mark=random_punct_mark(None), priority=0.6)
        )
        data.user['doingqp_happened'] = True

class WisdomQP(BehUnit):
    def analysis(self, data):
        return triggers_check(data.translated, lib['wisdom_qp']['triggers'])

    def generate(self):
        advice = choice(lib['wisdom_qp']['wisdom'])
        layout = choice(lib['wisdom_qp']['layouts'])
        return Chunk(
            layout.replace('*', advice)
        )
    
    def apply(self, data):
        data.answer.append(self.generate())

class FactsQP(BehUnit):
    def analysis(self, data):
        return triggers_check(data.translated, lib['facts_qp']['triggers'])
    
    def generate(self):
        fact = choice(lib['facts_qp']['facts'])
        layout = choice(lib['facts_qp']['layouts'])
        return Chunk(
            layout.replace('*', fact)
        )
    
    def apply(self, data):
        data.answer.append(self.generate())

class JokesQP(BehUnit):
    def analysis(self, data):
        return triggers_check(data.translated, lib['jokes_qp']['triggers'])
    
    def generate(self):
        joke = choice(lib['jokes_qp']['jokes'])
        layout = choice(lib['jokes_qp']['jokes'])
        return Chunk(
            layout.replace('*', joke), priority=0.8, mood=1.5
        )
    
    def apply(self, data):
        data.answer.append(self.generate())

class InterestQP(BehUnit):
    def __init__(self):
        self.collection = [
            WisdomQP,
            FactsQP,
            JokesQP
        ]
    
    def analysis(self, data):
        return triggers_check(data.translated, lib['interest_qp']['triggers'])
    
    def apply(self, data):
        author = choice(self.collection)
        generated = author.generate(None)
        data.answer.append(generated)

class BasicAnswersProcessor(BehUnit):
    def start(self):
        self.units = [
            StateQP(),
            NameQP(),
            DoingQP(),
            WisdomQP(),
            FactsQP(),
            JokesQP(),
            InterestQP()
        ]
    
    def on_new_userdata(self, userdata):
        for unit in self.units:
            unit.on_new_userdata(userdata)
        
    def apply(self, data):
        filtered = [unit for unit in self.units if unit.analysis(data)]
        if not filtered:
            return
        chosen = choice(filtered)
        chosen.apply(data)

class AddressingsAttacher(BehUnit):
    def apply(self, data):
        chunk = choice(data.answer)
        mood = data.user['mood']
        if mood >= 10:
            category = 'positive'
        elif mood <= -10:
            category = 'negative'
        else:
            category = 'neutral'
        addressing = choice(lib['addressings'][category])
        chunk.insert(addressing, position=randint(0, 1))

class RepeatingCatcher(BehUnit):
    def analysis(self, data):
        return flag(data.flags, 'repeating')
    def apply(self, data):
        main = random_line('spam')
        data.bot.apply_patience_to(-1.5, data.user)
        data.answer.append(
            Chunk(main, priority=0.75, mood=-1.5)
        )

class PrefixAttacher(BehUnit):
    def apply(self, data):
        chunk = choice(data.answer)
        prefix = choice(lib['prefixes'])
        chunk.insert(prefix, position=randint(0, 1))

class ImperativeAttacher(BehUnit):
    def apply(self, data):
        chunk = choice(data.answer)
        imp = choice(lib['imperative'])
        chunk.content = imp + ' ' + chunk.content

class RatingProcessor(BehUnit):
    def on_new_userdata(self, userdata):
        userdata['rating_requests'] = 0
        userdata['rating_history'] = []
    
    def analysis(self, data):
        data.user['rating_requests'] -= 1
        return True
    
    def disappointment(self, data, group):
        ans = choice(lib['rating_p'][group])
        data.bot.apply_patience_to(-1, data.user)
        data.answer.append(
            Chunk(ans, priority=0.85, mood=-1.2)
        )
    
    def is_repeating(self, subject, user):
        for subj in user['rating_history']:
            if string_similarity(subject, subj) >= 0.8:
                return True
        return False
    
    def apply(self, data):
        trigger = triggers_check(data.user_input, lib['rating_p']['triggers'])
        if not trigger:
            return
        data.user['rating_requests'] += 2
        if data.user['rating_requests'] >= 4:
            ans = choice(lib['rating_p']['overworking'])
            data.bot.apply_patience_to(-1, data.user)
            data.answer.append(
                Chunk(ans, priority=0.85, mood=-1)
            )
            return
        input_ = data.original_user_input.lower()
        end = re.search(trigger, input_).span()[1]
        try:
            space_id = input_[end:].index(' ') + end
        except:
            return
        stripped = input_[space_id + 1:]
        blob = TextBlob(stripped)
        subject_l = []

        for token in blob.tokens:
            if token in ['!', '?', '.']:
                break
            elif token in punctuation:
                continue
            subject_l.append(token)

        subject = ' '.join(subject_l)

        if self.is_repeating(subject, data.user):
            ans = choice(lib['rating_p']['repeating'])
            data.bot.apply_patience_to(-1, data.user)
            data.answer.append(
                Chunk(ans, priority=0.65, mood=-1.5)
            )
            return

        #print('SUBJECT:', subject)

        history = data.user['rating_history']
        history.append(subject)

        if len(history) > 50:
            history.pop(0)

        if conclude(0.25):
            factor = randint(-1, 0)
            ans = choice(lib['rating_p']['rejecting'])
        else:
            factor = randint(-1, 1)
            layout = choice(lib['rating_p']['layouts'])
            rating = random_rating(factor)
            ans = layout.replace('*', rating)

        data.answer.append(
            Chunk(ans, priority=0.75, mood=factor * rfloat(0.5, 2))
        )

class NewYearProcessor(BehUnit):
    def on_new_userdata(self, userdata):
        userdata['nyear_happened'] = False
    
    def analysis(self, data):
        return triggers_check(data.user_input, lib['new_year_p']['triggers'])
    
    def apply(self, data):
        if data.user['nyear_happened']:
            data.flags['repeating'] = True
            return
        starting = choice(lib['new_year_p']['startings'])
        wishes = random_wishes()

        data.answer += [
            Chunk(starting, mark=random_punct_mark('strong'), priority=1, mood=3.5),
            Chunk(wishes, mark=random_punct_mark('strong'), priority=0.8, mood=2.5)
        ]

        data.user['nyear_happened'] = True

class RandomAttacher(BehUnit):
    def analysis(self, data):
        return len(data.answer) > 0
    
    def apply(self, data):
        data.answer.append(
            Chunk(random_line('random'), priority=0.75)
        )

def random_rating(mood):
    zone = list(lib['rating']['words'].keys())[mood+ 1]
    rating = choice(lib['rating']['words'][zone])
    possible_amiplifiers = lib['rating']['amplifiers']
    if rating[0] == '*':
        border = len(possible_amiplifiers) - 1
        possible_amiplifiers = possible_amiplifiers[:border]
        rating = rating[1:]
    if conclude(0.6):
        amplifier = choice(possible_amiplifiers)
        rating = amplifier + ' ' + rating
    return rating

def random_wishes():
    layout = choice(lib['wishes']['layouts'])
    count = randint(1, len(lib['wishes']['things']) // 1.5)
    wishes_l = rshuffle(lib['wishes']['things'])[:count]
    if count == 1:
        wishes = wishes_l[0]
    else:
        wishes = ', '.join(wishes_l[:count - 1]) + ' и ' + wishes_l[-1]
    return layout.replace('*', wishes)

class Bot:
    def __init__(self, behaviour, name='Bot'):
        self.beh = behaviour
        self.name = name

    def apply_mood_to(self, velocity, user):
        user['mood'] += velocity * rfloat(0.75, 1.25)

    def apply_patience_to(self, velocity, user):
        user['patience'] += velocity

    def disconnect_from(self, userdata):
        userdata['forbidden'] = True

    def init_userdata(self, userdata):
        for unit in self.beh:
            unit.on_new_userdata(userdata)
        userdata.initialized = True

    def on_error(self, data):
        line = random_line('empty')
        mark = random_punct_mark('weak')
        # ...

    def passthrough(self, data):
        try:
            runthrough(self.beh, data)
            return data.answer
        except Exception as ex:
            if debug:
                print('Error:', ex)
            self.on_error(data)

    def generate_outdialog_message(self):
        return 'OUTDIALOG MESSAGE'

    def get_component(self, type_):
        for comp in self.beh:
            if type(comp) == type_:
                return comp
        return None

print('LOADING BOT COMPONENTS...')

bot = Bot([
    SpamHandler(),
    Parting(),
    Greeting(),
    BasicAnswersProcessor(),
    DirtySlangCatcher(chance=0.5),
    InputMoodProcessor(), 
    RatingProcessor(),
    NewYearProcessor(),
    #Neural(),
    Alternative(),
    InsultAttacher(chance=0.5),
    RandomAttacher(chance=0.1),
    RepeatingCatcher(),
    Protector(),
    OrderRandomizer(units=[
        LaughAttacher(chance=0.05),
        PrefixAttacher(chance=0.1),
        ImperativeAttacher(chance=0.1),
        AddressingsAttacher(chance=0.1)
    ]),
    ChunksMoodCounter(),
    InnerMoodProcessor()
], name='Савелий')

print('DONE!')

if __name__ == '__main__':
    userdata = UserData()
    while (inp := input('> ')) != '':
        if not bot.online:
            continue
        print(bot.name + ':', bot.answer(inp, userdata))