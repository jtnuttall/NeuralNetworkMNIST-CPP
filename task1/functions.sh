# diff with width $1, lines $2, file 1 $3, file 2 $4, cut word $5
diff_results() {
	IFS=
	f1="$(head -n $2 $3 | sed "s/^.*$5/$5/" | nl)"
	f2="$(head -n $2 $4 | sed "s/^.*$5/$5/" | nl)"
	out="diff"$3"-"$4
	diff -y -W $1 <(echo $f1) <(echo $f2) > $out
	cat $out
}