def pretty_time_delta(seconds: float, seconds_fmt: str = "%.1f") -> str:
    sign_string = "-" if seconds < 0 else ""

    seconds = abs(seconds)
    microseconds = seconds - int(seconds)
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    ret = sign_string
    if days > 0:
        ret += f"{days}d"
    if hours > 0:
        ret += f"{hours}h"
    if minutes > 0:
        ret += f"{minutes}m"

    seconds = seconds + microseconds
    ret += (seconds_fmt + "s") % seconds
    return ret
