import os

from website import create_app

app = create_app()

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(
        debug=debug,
        use_reloader=debug,
        host="127.0.0.1",
        port=int(os.getenv("PORT", 5000)),
    )
