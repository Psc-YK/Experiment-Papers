import logging
import datetime


def get_logger(config):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    plt_name = str(config['dataset']) + ' ' + str(config['missing_rate']).replace('.','') + ' ' + str(
        datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S'))
    fh = logging.FileHandler(
        './logs/' + str(config['dataset']) + ' ' + str(config['missing_rate']).replace('.','') + ' ' + str(
            datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')) + '.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, plt_name
# get_logger 函数接受一个 config 对象作为输入，并返回一个日志记录器对象和一个用于绘图的名称。
# 函数首先将 matplotlib.font_manager 的日志级别设置为 WARNING，以抑制不必要的日志消息。
# 然后，它创建一个日志记录器对象，并将其日志级别设置为 DEBUG。
# 接下来，它创建一个格式化器对象，用于设置日志消息的特定格式。格式化器的格式包括时间戳、名称、日志级别和消息。
# 函数接着创建一个文件处理器对象，将日志消息写入文件。文件名基于 config 对象中的 dataset 和 missing_rate 值以及当前的日期和时间生成。
# 文件处理器的日志级别设置为 DEBUG，并配置为使用先前创建的格式化器。
# 同时，函数还创建一个流处理器对象，用于将日志消息输出到控制台。流处理器的日志级别设置为 DEBUG，并配置为使用相同的格式化器。
# 最后，文件处理器和流处理器被添加到日志记录器中，并返回日志记录器和绘图名称。


