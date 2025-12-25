#!/usr/bin/env perl
# Copyright 2025, author: Ibrahim Almajai         
# Apache 2.0

while(<>){
    m:.*/([^/]+)\.wav$: || die "Bad line $_";
    $id = $1;
    $id =~ tr/A-Z/a-z/;  # Lowercase the id
    $id =~ tr/_/-/;
    $id =~ tr/(//d;
    $id =~ tr/)//d;
    print "$id $_";
}



