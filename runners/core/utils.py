from datetime import datetime
from dateutil.relativedelta import relativedelta


def generate_date_lists(initial_config, n):
    generated_lists = {}

    for name, start_date_str in initial_config.items():
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        date_list = []

        for m in range(n):
            new_date = start_date + relativedelta(months=m)
            date_list.append(new_date.strftime("%Y-%m-%d"))

        list_name = f"{name}_list"
        generated_lists[list_name] = date_list

    return generated_lists
