#!/usr/bin/perl

while (<>)
{
    chop;

    if ( m/\blab\b/ || m/\brec\b/ ) 
    {
        $n = 0;
        $seg = $_;
    }
    elsif ( m/^[.]/ )
    {
        if ( $n == 0 )
        {
            $seg =~ s/\"//g;
            @p = split(/\//,$seg);
            print "$p[$#p]\n";
        }
    }
    else
    {
        $n++;
    }
}
