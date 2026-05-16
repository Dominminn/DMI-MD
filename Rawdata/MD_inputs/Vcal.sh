#!/bin/sh
STEP=17
for i in $(seq 32 -2 0); do
  dir="${i}Na_tip3p"
  if [ -d "$dir" ]; then
    cp compute_drop.py "$dir"/
    ( cd "$dir" && \
      echo "$dir" && \
      ~/scripts/cube_add o_"$STEP".cube h_"$STEP".cube ; \
      ~/scripts/cube_add add.cube ion_"$STEP".cube ; \
      mv add.cube md.cube && \
      cp md.cube md_"$STEP".cube && \
      python3 compute_drop.py --cube "md.cube" \
      --mirror "12.65456957 1.587731995 8.890343301 11.61048672 1.455525298 8.890343301 4.228545984 4.395561901" >> ../temp.txt )
  fi
done

