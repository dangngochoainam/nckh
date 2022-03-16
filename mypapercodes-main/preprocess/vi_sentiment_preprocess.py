import re
import utils


def remove_repeated_characters(text):
    def rep(obj):
        idx = obj.string.index(obj.group(0))
        if idx > 0:
            pre = remove_vietnamese_accents(obj.string[idx - 1])
            if obj.group(0)[0] == pre:
                return ''

        return obj.group(1)

    return re.sub(r'([A-Z])\1+', rep, text, flags=re.IGNORECASE)


def remove_vietnamese_accents(text):
    result = ''
    for c in text:
        result += utils.s0[utils.s1.index(c)] if c in utils.s1 else c

    return result


def handle_negation_form(text):
    text = re.split("\s*[\s,;]\s*", text)

    for idx in range(len(text)):
        if idx < len(text) - 1 and text[idx] in utils.not_list:
            if text[idx + 1] in utils.pos_list:
                text[idx] = "notpositive"
                text[idx + 1] = ""
            if text[idx + 1] in utils.neg_list:
                text[idx] = "notnegative"
                text[idx + 1] = ""
        elif text[idx] not in utils.not_list:
            if text[idx] in utils.pos_list:
                text.append("positive")
            elif text[idx] in utils.neg_list:
                text.append("negative")

    return " ".join(text)
