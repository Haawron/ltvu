import sys
import logging
from sty import fg, ef, rs, Style, RgbFg

fg.orange = Style(RgbFg(255, 150, 50))


class RelativeTimeFormatter(logging.Formatter):
    def format(self, record):
        relative_seconds = record.relativeCreated / 1000.0
        hours, remainder = divmod(relative_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        record.relative_time = "%02d:%02d:%02d" % (hours, minutes, seconds)
        return super().format(record)


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all logs
    datefmt = '%Y-%m-%d %H:%M:%S'

    debug_snippet = fg.orange + ef.bold + 'DEBUG' + rs.all
    debug_handler = logging.StreamHandler(sys.stderr)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
    # TODO: module(short) vs. name(long)
    debug_format = '%(relative_time)s | ' + debug_snippet + ' | %(module)s | %(message)s'
    debug_format = RelativeTimeFormatter(debug_format, datefmt=datefmt)
    debug_handler.setFormatter(debug_format)

    main_handler = logging.StreamHandler(sys.stdout)
    main_handler.setLevel(logging.INFO)
    main_handler.addFilter(lambda record: record.levelno == logging.INFO)
    main_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
    main_format = logging.Formatter(main_format, datefmt=datefmt)
    main_handler.setFormatter(main_format)

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.WARNING)  # or above
    error_handler.setFormatter(main_format)

    logger.addHandler(debug_handler)
    logger.addHandler(main_handler)
    logger.addHandler(error_handler)

    return logger


if __name__ == '__main__':
    logger = logging.getLogger('LTVU')
    logger.debug('test setup_logger')
    logger.info('test setup_logger')
    logger.warning('test setup_logger')
    logger.error('test setup_logger')
    logger.critical('test setup_logger')
