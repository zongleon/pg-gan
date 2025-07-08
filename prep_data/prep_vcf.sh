#!/bin/bash

PREFIX=/bigdata/smathieson/1000g-share
SUFFIX=.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz

# for each set of population(s)
for POP in CHB
do

  # for each chromosome
  for CHROM in `seq 1 22`
  do
      #echo "bcftools view -S /homes/smathieson/Public/1000g/sample_lists/${POP}_all.txt --min-ac 1:minor -m2 -M2 -v snps -Oz -o ${PREFIX}/VCF\
      #/${POP}_h108.chr${CHROM}${SUFFIX} ${PREFIX}/ALL/ALL.chr${CHROM}${SUFFIX}"
      bcftools view -S /homes/smathieson/Public/1000g/sample_lists/${POP}_all.txt --min-ac 1:minor -m2 -M2 -v snps -Oz -o ${PREFIX}/VCF/${POP}_h103.chr${CHROM}${SUFFIX} ${PREFIX}/ALL/ALL.chr${CHROM}${SUFFIX}
  done

  # then merge into one vcf
  echo "bcftools concat -f ${POP}_h103_filelist.txt -Oz -o ${PREFIX}/VCF/${POP}_h103${SUFFIX}"
  bcftools concat -f ${POP}_h103_filelist.txt -Oz -o ${PREFIX}/VCF/${POP}_h103${SUFFIX}
done
