#!/usr/bin/env perl
use strict;
use warnings FATAL => "all";

# make sure to set VALIDATION_MODE = 1 in main.cpp
# usage: ./task3 | ./tocsv.pl <outputFile>
# output: 1 csv file, with <epoch>,<trainingLoss>,<validationLoss> on each line

my $outName = shift @ARGV;
open my $outFile, '>', $outName or die "Could not open file " . $outName;

print {$outFile} "epoch,trainingLoss,validationLoss\n";

while (<>) {
	next if not (/^epoch/);
	my @cols = /[-+]?[0-9]*\.?[0-9]+/g;
	my $line = $cols[0].",".$cols[2].",".$cols[4]."\n";
	print {$outFile} $line;
}