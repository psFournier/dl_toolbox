cd "${TMPDIR}"

mkdir miniworld_tif
cities=( "christchurch" )
for city in "${cities[@]}" ; do
    rsync -rv --include="${city}/" /work/OT/ai4usr/fournip/miniworld_tif/ miniworld_tif/
done
