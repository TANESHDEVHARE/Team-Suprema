NAME          TINYUNB
* min -x1 - x2  s.t.  x1 - x2 <= 1,  x1 + x2 >= 1,  x >= 0 : unbounded along (1,1)
ROWS
 N  COST
 L  R1
 G  R2
COLUMNS
    X1        COST      -1.0       R1        1.0
    X1        R2        1.0
    X2        COST      -1.0       R1        -1.0
    X2        R2        1.0
RHS
    RHS       R1        1.0        R2        1.0
ENDATA
