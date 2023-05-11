cd "${TMPDIR}"
mkdir DIGITANIE
#cities=( "Arcachon" "Biarritz" "Montpellier" "Toulouse" "Nantes" "Strasbourg" "Paris" )
cities=( )
for city in "${cities[@]}" ; do
    #rsync -rv --include="${city}/" --include="COS9/" --include="*.tif" --exclude="*" /work/OT/ai4geo/DATA/DATASETS/DIGITANIE/ DIGITANIE/
    rsync -rvL --include="${city}/" --include="COS9/" --include="*.tif" --exclude="*" /work/OT/ai4geo/DATA/DATASETS/DIGITANIE/ DIGITANIE/
done
