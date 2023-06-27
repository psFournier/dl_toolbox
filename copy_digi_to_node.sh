cd "${TMPDIR}"

mkdir DIGITANIE_v3
cities=( "ARCACHON" "BIARRITZ" "MONTPELLIER" "TOULOUSE" "NANTES" "STRASBOURG" "PARIS" "BRISBANE" "BUENOS-AIRES" "CAN-THO" "HELSINKI" "LAGOS" "LE-CAIRE" "MAROS" "MUNICH" "NEW-YORK" "PORT-ELISABETH" "RIO-JANEIRO" "SAN-FRANCISCO" "SHANGHAI" "TIANJIN" )
#cities=( "Toulouse" )
for city in "${cities[@]}" ; do
    rsync -rv --include="${city}/" --include="COS9/" --include="*.tif" --exclude="*" /work/OT/ai4geo/DATA/DATASETS/DIGITANIE_v3/ DIGITANIE_v3/
done