import os

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity

from openai_utils import OpenAiUtils

EXECUTION_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXECUTION_DIR_PATH, "data")
IMAGE_VECTORS_PATH = os.path.join(DATA_DIR, "image_vectors.npy")
KEYWORDS_CSV_PATH = os.path.join(EXECUTION_DIR_PATH, "data/keywords.csv")
MAX_RESUlT_IMAGE_NUM = 10

app = Flask(__name__)
CORS(app, origins=["http://localhost:8080"])


@app.route("/search", methods=["POST"])
def search_picto() -> Response:
    if not request.json:
        return jsonify(
            {"success": False, "message": "Query is empty.", "pictId": "9999"}
        )
    query = request.json.get("query")
    print(f"query: {query}")

    openaiutils = OpenAiUtils()
    text_vec = openaiutils.do_embedding(query)
    if text_vec is None:
        return jsonify(
            {
                "success": False,
                "message": "Failed do_embedding.",
                "pictId": "9999",
            }
        )

    image_vectors = np.load(IMAGE_VECTORS_PATH)  # type: ignore
    similarities = cosine_similarity([text_vec], image_vectors)
    ids = similarities.argsort()[0][-MAX_RESUlT_IMAGE_NUM:]

    df_keywords = pd.read_csv(KEYWORDS_CSV_PATH)

    return jsonify(
        {
            "success": True,
            "message": "Search pictogram successfully",
            "pictoId": df_keywords.loc[ids]["id"].values.tolist(),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
