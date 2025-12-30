#!/usr/bin/env perl
# Copyright   2025  (author: Ibrahim Almajai)         
# Apache 2.0

while(<>){
    m:^\S+/(\S+)/(\S+)\.mpg$: ;
    $dir = $1;
    $id = $2;
    $dir =~ tr/A-Z/a-z/;
    $id =~ tr/A-Z/a-z/;
    print "$dir-$id $_";
}

