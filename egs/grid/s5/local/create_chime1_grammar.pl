#!/usr/bin/env perl
#
# copyright 2015  university of sheffield (author: ning ma)
# apache 2.0.
#
# prepare a simple grammar g.fst for the grid corpus (chime 1/2)
# with silence at the beginning and the end of each utterance.
#

use strict;
use warnings;

# grid has the following grammar:
# verb=bin|lay|place|set
# colour=blue|green|red|white
# prep=at|by|in|with
# letter=a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|x|y|z
# digit=zero|one|two|three|four|five|six|seven|eight|nine
# coda=again|now|please|soon
# sil $verb $colour $prep $letter $digit $coda sil

my $state = 0;
my $state2 = $state + 1;
#my $sil = "<sil>";
#print "$state $state2 $sil $sil 0.0\n";

#$state++;
#$state2 = $state + 1;
my @words = ("bin", "lay", "place", "set");
my $nwords = @words;
my $penalty = -log(1.0/$nwords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

$state++;
$state2 = $state + 1;
@words = ("blue", "green", "red", "white");
$nwords = @words;
$penalty = -log(1.0/$nwords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

$state++;
$state2 = $state + 1;
@words = ("at", "by", "in", "with");
$nwords = @words;
$penalty = -log(1.0/$nwords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

$state++;
$state2 = $state + 1;
@words = ("a".."v", "x", "y", "z");
$nwords = @words;
$penalty = -log(1.0/$nwords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

$state++;
$state2 = $state + 1;
@words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine");
$nwords = @words;
$penalty = -log(1.0/$nwords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

$state++;
$state2 = $state + 1;
@words = ("again", "now", "please", "soon");
$nwords = @words;
$penalty = -log(1.0/$nwords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

#$state++;
#$state2 = $state + 1;
#print "$state $state2 $sil $sil 0.0\n";

print "$state2 0.0\n";

