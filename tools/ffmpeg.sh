ffmpeg -r 12 -i plot%05d.png -vf scale=2048:1800 -qscale:v 0 -r 12 movie.avi
