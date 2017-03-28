echo 'Installling icon...'

echo 'creating symbolic links.'

# make sure the base
if [ ! -d "code/web/resources" ]; then
    echo "directory code/web/resources is missing."
    exit
fi

cd code/web/resources
rm output labels train validate
ln -s ../../../data/train train
ln -s ../../../data/valid validate
ln -s ../../../data/segmentation output
ln -s ../../../data/labels labels
cd ../../..

echo 'creating database.'
cd code/database
python setup.py

cd ../common
#python performance.py install
cd ../..

