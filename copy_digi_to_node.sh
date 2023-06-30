cd "${TMPDIR}"

mkdir DIGITANIE_v3
cd DIGITANIE_v3
cities=( "ARCACHON" "BIARRITZ" "MONTPELLIER" "TOULOUSE" "NANTES" "STRASBOURG" "PARIS" "BRISBANE" "BUENOS-AIRES" "CAN-THO" "HELSINKI" "LAGOS" "LE-CAIRE" "MAROS" "MUNICH" "NEW-YORK" "PORT-ELISABETH" "RIO-JANEIRO" "SAN-FRANCISCO" "SHANGHAI" "TIANJIN" )
cities=( "TOULOUSE" )
for city in "${cities[@]}" ; do
#    rsync -rv --include="${city}/" --include="COS9/" --include="*.tif" --exclude="*" /work/OT/ai4geo/DATA/DATASETS/DIGITANIE_v3/ DIGITANIE_v3/
    mkdir -p ${city}/COS9
    cp /work/AI4GEO/data/DATA/DATASETS/DIGITANIE_v3/${city}/*cog_tile_*.tif ${city}/
    cp /work/AI4GEO/data/DATA/DATASETS/DIGITANIE_v3/${city}/COS9/*mask_cog.tif ${city}/COS9/
done