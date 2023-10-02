cd "${TMPDIR}"
mkdir DIGITANIE_v4
cd DIGITANIE_v4
cities=( "ARCACHON" "BIARRITZ" "MONTPELLIER" "TOULOUSE" "NANTES" "STRASBOURG" "PARIS" "BRISBANE" "BUENOS-AIRES" "CAN-THO" "HELSINKI" "LAGOS" "LE-CAIRE" "MAROS" "MUNICH" "NEW-YORK" "PORT-ELISABETH" "RIO-JANEIRO" "SAN-FRANCISCO" "SHANGHAI" "TIANJIN" )
cities=( "TOULOUSE" )
cp /work/AI4GEO/data/DATA/DATASETS/DIGITANIE_v4/normalisation_stats.npy .
for city in "${cities[@]}" ; do
    cp -rd /work/AI4GEO/data/DATA/DATASETS/DIGITANIE_v4/${city} .
#    rsync -rv --include="${city}/" --include="COS9/" --include="*.tif" --exclude="*" /work/OT/ai4geo/DATA/DATASETS/DIGITANIE_v3/ DIGITANIE_v3/
done