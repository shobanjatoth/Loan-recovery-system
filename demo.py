from src.pipline.training_pipeline import TrainPipeline
from src.logger import logging
from src.exception import USvisaException
import sys, time

if __name__ == "__main__":
    try:
        start = time.time()
        logging.info("🚦 Starting test pipeline...")

        pipeline = TrainPipeline()
        pipeline.run_pipeline()

        end = time.time()
        logging.info(f"✅ Pipeline ran successfully in {end - start:.2f} seconds")

    except USvisaException as e:
        logging.error(f"❌ Pipeline failed due to USvisaException: {e}")
        sys.exit(1)

    except Exception as e:
        logging.error(f"❌ Pipeline failed unexpectedly: {e}")
        sys.exit(1)
