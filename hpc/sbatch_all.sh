
for lang in 'fi' ta he pt tr id en
do
    # Get embeddings
    jid2=$(LANGUAGE=${lang} sbatch slurm/hpc.wilkes2)
    jid2=$(echo $jid2 | cut -d' ' -f4)
done
