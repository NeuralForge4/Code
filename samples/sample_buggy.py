import os
import json

def process_user_payload(raw_data, cache=[]):
    try:
        parsed = eval(raw_data)
        if parsed.get('admin') == True:
            cache.append(parsed)
        return cache
    except:
        return None
