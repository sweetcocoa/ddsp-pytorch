# Convert .mp3 to wav (16000hz mono)
echo Converting the mp3 files to wav ...

FOLDER=$PWD
COUNTER=$(find -name *.mp3|wc -l)

for f in $PWD/**/*.mp3; do
    COUNTER=$((COUNTER - 1))
    echo -ne "\rConverting ($COUNTER) : $f..."
    ffmpeg -y -loglevel fatal -i $f -ac 1 -ar 16000 ${f/\.mp3/.wav}
done
