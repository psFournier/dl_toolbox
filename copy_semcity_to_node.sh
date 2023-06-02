cd "${TMPDIR}"

mkdir SemCity-Toulouse-bench
rsync -rv --include="img_multispec_05/" --include="TLS_BDSD_M/" --include="*.tif" --exclude="*" /work/OT/ai4usr/fournip/SemCity-Toulouse-bench/ SemCity-Toulouse-bench/
rsync -rv --include="semantic_05/" --include="TLS_GT/" --include="*1.tif" --exclude="*" /work/OT/ai4usr/fournip/SemCity-Toulouse-bench/ SemCity-Toulouse-bench/