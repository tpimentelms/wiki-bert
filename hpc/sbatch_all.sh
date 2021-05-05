
for lang in 'fi' ta he pt tr id en
do
    # Get embeddings
    jid2=$(LANGUAGE=${lang} sbatch hpc/hpc.wilkes2)
    jid2=$(echo $jid2 | cut -d' ' -f4)
    echo "Submitted " ${lang} ${jid2}
    LANGUAGE=${lang} sbatch --dependency=afternotok:${jid2} hpc/hpc.wilkes2
done
