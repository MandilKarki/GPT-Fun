"""
-- Created by: Ashok Kumar Pant
-- Created on: 10/19/21
"""
import requests


def test_api_running():
    context = "test"
    payload = {
        "context": context,
        "length": 2,
        "temp": 0.0,
        "top_p": 0.0,
    }
    response = requests.post("http://localhost:8080/generate", params=payload).json()
    print(response)
    assert len(response["text"]) > 0


if __name__ == '__main__':
    test_api_running()
