QWEN_LANG_MAP = {
    "zh":"Chinese","en":"English","fr":"French",
    "es":"Spanish","pt":"Portuguese","de":"German",
    "it":"Italian","ru":"Russian","ja":"Japanese",
    "ko":"Korean","vi":"Vietnamese","th":"Thai",
    "ar":"Arabic"
                 }

WHISPER_LANG_MAP = {
    "en":"English","zh":"Chinese","de":"German",
    "es":"Spanish","ru":"Russian","ko":"Korean",
    "fr":"French","ja":"Japanese","pt":"Portuguese"
    ,"tr":"Turkish","pl":"Polish","ca":"Catalan",
    "nl":"Dutch","ar":"Arabic","sv":"Swedish",
    "it":"Italian","id":"Indonesian","hi":"Hindi"
    ,"fi":"Finnish","vi":"Vietnamese","he":"Hebrew"
    ,"uk":"Ukrainian","el":"Greek","ms":"Malay",
    "cs":"Czech","ro":"Romanian","da":"Danish",
    "hu":"Hungarian","ta":"Tamil","no":"Norwegian"
    ,"th":"Thai","ur":"Urdu","hr":"Croatian","bg":"Bulgarian"
    ,"lt":"Lithuanian","la":"Latin","mi":"Maori","ml":"Malayalam"
    ,"cy":"Welsh","sk":"Slovak","te":"Telugu","fa":"Persian"
    ,"lv":"Latvian","bn":"Bengali","sr":"Serbian","az":"Azerbaijani",
    "sl":"Slovenian","kn":"Kannada","et":"Estonian","mk":"Macedonian"
    ,"br":"Breton","eu":"Basque","is":"Icelandic","hy":"Armenian",
    "ne":"Nepali","mn":"Mongolian","bs":"Bosnian","kk":"Kazakh",
    "sq":"Albanian","sw":"Swahili","gl":"Galician","mr":"Marathi",
    "pa":"Punjabi","si":"Sinhala","km":"Khmer","sn":"Shona",
    "yo":"Yoruba","so":"Somali","af":"Afrikaans","oc":"Occitan",
    "ka":"Georgian","be":"Belarusian","tg":"Tajik","sd":"Sindhi",
    "gu":"Gujarati","am":"Amharic","yi":"Yiddish","lo":"Lao",
    "uz":"Uzbek","fo":"Faroese","ht":"Haitian Creole","ps":"Pashto",
    "tk":"Turkmen","nn":"Norwegian Nynorsk","mt":"Maltese","sa":"Sanskrit",
    "lb":"Luxembourgish","my":"Myanmar (Burmese)","bo":"Tibetan","tl":"Tagalog",
    "mg":"Malagasy","as":"Assamese","tt":"Tatar","haw":"Hawaiian","ln":"Lingala",
    "ha":"Hausa","ba":"Bashkir","jw":"Javanese",
    "su":"Sundanese","yue":"Cantonese"
                    }

# {en: english} 형식으로 저장
COMMON_LANG_MAP = {
            code: WHISPER_LANG_MAP[code]
            for code in WHISPER_LANG_MAP
            if code in QWEN_LANG_MAP
        }
        