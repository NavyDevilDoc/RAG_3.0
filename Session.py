from datetime import datetime
import json
import uuid

class Session:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.query_count = 0
        
    def to_json(self):
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "query_count": self.query_count
        }
