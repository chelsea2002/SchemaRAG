GPT_CHAT_MODELS = ['gpt-3.5-turbo', 'gpt-4o-mini']

GPT_COMPLETION_MODELS = ['code-davinci-002', 'text-davinci-003']

MAX_LENS = {
    'gpt-3.5-turbo': 7500,
    'gpt-4o-mini': 15000,
    'code-davinci-002': 15000,
    'text-davinci-003': 7500
}

AGGS = [None, 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

CONDS = [None, 'BETWEEN', '=', '>', '<', '>=', '<=', '!=', 'IN', 'LIKE']

OPS = [None, '-', '+', '*', '/']

SET_OPS = ['intersect', 'union', 'except']

