import logging

# Set up global logging configuration to log to a file
logging.basicConfig(filename='main_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
main_logger = logging.getLogger(__name__)

def setup_logger(log_filename):
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def some_function():
    # Use the main logger
    main_logger.info("Running some_function")

    # Set up a specific logger for detailed logs
    specific_logger = setup_logger('specific_log.log')
    specific_logger.info("Detailed log message from some_function")

# Initialize the main logger
main_logger.info("Starting the script")

# Call the function
some_function()
