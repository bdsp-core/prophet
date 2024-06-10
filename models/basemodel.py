import os
import subprocess
import logging
import json
import re

logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, config):
        logger.info("Initializing BaseModel")
        self.config = config
        self._initialize()

    def _initialize(self):
        pass
        # env = self.config['environment']
        # if os.path.exists(env['path']):
        #     logger.info(f"Environment already exists: {env['path']}")
        # else:
        #     logger.info(f"Environment not found: {env['path']}")
        #     logger.info(f"Creating environment: {env['name']}")
        #     result = subprocess.run(['conda', 'create', '-f', env['dependencies_path'], '--name', env['name'], '-y'], capture_output=True)
        #     if result.returncode != 0:
        #         logger.error(f"Failed to create environment: {env['name']}. Error: {result.stderr.decode('utf-8')}")
        #         raise ValueError(f"Failed to create environment: {env['name']}. Error: {result.stderr.decode('utf-8')}")
        #     else:
        #         logger.info(f"Environment created: {env['name']}")
        #         env['path'] = re.search(r'environment location: (.*)\b', result.stdout.decode('utf-8'))[1]
        # TODO: venv
