PROMPT = "Translate the given sentence into {}"

# Use mBART language tags, which are not uniformly <language>_XX.
# Unsupported languages are rejected instead of silently becoming English.
_LANGUAGE_CODES = {
    "english": "en_XX",
    "chinese": "zh_CN",
    "german": "de_DE",
    "french": "fr_XX",
    "spanish": "es_XX",
    "italian": "it_IT",
    "portuguese": "pt_XX",
    "dutch": "nl_XX",
    "russian": "ru_RU",
    "japanese": "ja_XX",
    "korean": "ko_KR",
    "arabic": "ar_AR",
    "hindi": "hi_IN",
    "thai": "th_TH",
    "vietnamese": "vi_VN",
    "indonesian": "id_ID",
    "turkish": "tr_TR",
    "polish": "pl_PL",
    "czech": "cs_CZ",
    "romanian": "ro_RO",
    "finnish": "fi_FI",
    "swedish": "sv_SE",
    "hebrew": "he_IL",
    "ukrainian": "uk_UA",
    "croatian": "hr_HR",
    "slovenian": "sl_SI",
    "estonian": "et_EE",
    "latvian": "lv_LV",
    "lithuanian": "lt_LT",
}
LANG_MAP = {
    **_LANGUAGE_CODES,
    **{code.split("_")[0]: code for code in _LANGUAGE_CODES.values()},
    **{code.lower(): code for code in _LANGUAGE_CODES.values()},
}
