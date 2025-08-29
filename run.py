import os

from website import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG") == "1",
        use_reloader=False,
        threaded=True,
        host="127.0.0.1",
        port=5000,
    )
