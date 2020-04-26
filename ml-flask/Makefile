

install:
	- virtualenv -p python3 ../ml-flask/build
	- ( \
       . build/bin/activate; \
        pip3 install -r requirements.txt; \
    )
run:
	- ( \
       . build/bin/activate; \
	   FLASK_APP=app.py flask run\
    )

@test:
	- ( \
       . build/bin/activate; \
	   pytest sre/test/features/unit/test_dataset.py\
    )