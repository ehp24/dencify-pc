Project to densify 3D road point clouds using denser road 2D RGB images and camera lidar fusion

This script main.py when run on venv conatining necessary dependencies will return a densified point cloud las object in LAS folder
Have broken down functions into 3 modules within the densify_pc package, dataprocessor.py, projection.py and utils.py, and i have placed my functions into the module depending on the fucntionality of the function.

toml file and lock file are used to create virtual env with poetry

I have written unit tests files for modules dataprocessing and utils but NOT projection.

__init__.py inside the test file and densify_pc package just mark these folders as packages - this is important. Though they are empty now which is fine.

1. initiate poetry environement wiht poetry init and poetry install if starting fresh
2. activate the venv using `poetry shell` whilst inside the densify-pc directory
3. run the code by `python main.py` with the poetry venv activated
4. Should take aournd 100seconds to process one image. Make sure you have an image in data>raw>img and LAS inside data>raw>las and csv in data>raw>csv. The las needs to correpond to data int he csv, and the img also needs to occur in the csv and in the las highway section.
5. to run pytest unit tests, type `pytest` and this will run unit tests inside test folder

NOTE: there are still lots of unfinished thinsg e.g. unit tests for projection, interp unit test should test if we have correct values interpolated 
* Missing test_projection.py unit tests
* Some docstrings are outdated/ have not been fully completed and i did not change them in time
* should add type hinting in function arg declaration
* pytest for inetrpolate function inside data processor shuld check the inetrpolated values are correct