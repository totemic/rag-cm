# Colbert + DB retrieval
## 1 concurrent request
% ab -n 1000 -c 1 "http://127.0.0.1:8000/query?q=test?&count=10" 
Document Path:          /query?q=test?&count=10
Document Length:        4004 bytes

Concurrency Level:      1
Time taken for tests:   39.594 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      4131000 bytes
HTML transferred:       4004000 bytes
Requests per second:    25.26 [#/sec] (mean)
Time per request:       39.594 [ms] (mean)
Time per request:       39.594 [ms] (mean, across all concurrent requests)
Transfer rate:          101.89 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.0      0       0
Processing:    36   39   3.5     39      85
Waiting:       36   39   3.4     39      85
Total:         37   40   3.5     39      85

Percentage of the requests served within a certain time (ms)
  50%     39
  66%     39
  75%     40
  80%     40
  90%     42
  95%     43
  98%     44
  99%     49
 100%     85 (longest request)

## 10 concurrent requests
% ab -n 1000 -c 10 "http://127.0.0.1:8000/query?q=test?&count=10"
Document Path:          /query?q=test?&count=10
Document Length:        4004 bytes

Concurrency Level:      10
Time taken for tests:   41.760 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      4131000 bytes
HTML transferred:       4004000 bytes
Requests per second:    23.95 [#/sec] (mean)
Time per request:       417.600 [ms] (mean)
Time per request:       41.760 [ms] (mean, across all concurrent requests)
Transfer rate:          96.60 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.1      0       1
Processing:    69  415  30.3    418     504
Waiting:       69  415  30.3    418     503
Total:         70  415  30.3    418     504

Percentage of the requests served within a certain time (ms)
  50%    418
  66%    423
  75%    426
  80%    428
  90%    433
  95%    443
  98%    480
  99%    482
 100%    504 (longest request)


# Sync DB retrieval only (return results from sqlite only, no Colbert)

## 1 concurrent request
% ab -n 1000 -c 1 "http://127.0.0.1:8000/querytest"
Document Path:          /querytest
Document Length:        3419 bytes

Concurrency Level:      1
Time taken for tests:   0.967 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      3546000 bytes
HTML transferred:       3419000 bytes
Requests per second:    1033.63 [#/sec] (mean)
Time per request:       0.967 [ms] (mean)
Time per request:       0.967 [ms] (mean, across all concurrent requests)
Transfer rate:          3579.36 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.0      0       1
Processing:     1    1   0.5      1       6
Waiting:        0    1   0.4      1       5
Total:          1    1   0.6      1       7

Percentage of the requests served within a certain time (ms)
  50%      1
  66%      1
  75%      1
  80%      1
  90%      1
  95%      2
  98%      3
  99%      4
 100%      7 (longest request)

## 10 concurrent requests
% ab -n 1000 -c 10 "http://127.0.0.1:8000/querytest"    
Document Path:          /querytest
Document Length:        3419 bytes

Concurrency Level:      10
Time taken for tests:   0.774 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      3546000 bytes
HTML transferred:       3419000 bytes
Requests per second:    1292.18 [#/sec] (mean)
Time per request:       7.739 [ms] (mean)
Time per request:       0.774 [ms] (mean, across all concurrent requests)
Transfer rate:          4474.68 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.1      0       3
Processing:     3    8   1.8      7      25
Waiting:        3    7   1.8      7      25
Total:          4    8   1.8      7      25

Percentage of the requests served within a certain time (ms)
  50%      7
  66%      7
  75%      8
  80%      8
  90%      9
  95%     10
  98%     13
  99%     17
 100%     25 (longest request)

# 100 concurrent requests
% ab -n 1000 -c 100 "http://127.0.0.1:8000/querytest"    
Document Path:          /querytest
Document Length:        3419 bytes

Concurrency Level:      100
Time taken for tests:   0.761 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      3546000 bytes
HTML transferred:       3419000 bytes
Requests per second:    1314.40 [#/sec] (mean)
Time per request:       76.081 [ms] (mean)
Time per request:       0.761 [ms] (mean, across all concurrent requests)
Transfer rate:          4551.61 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   1.0      0       4
Processing:     7   72   8.2     71      99
Waiting:        3   72   8.3     71      99
Total:          7   72   8.1     71     102

Percentage of the requests served within a certain time (ms)
  50%     71
  66%     72
  75%     74
  80%     75
  90%     82
  95%     87
  98%     89
  99%     96
 100%    102 (longest request)

# Async DB only, no connection re-use
## 1 concurrent request
% ab -n 1000 -c 1 "http://127.0.0.1:8000/queryaiotest?q=test"
Document Path:          /queryaiotest?q=test
Document Length:        3569 bytes

Concurrency Level:      1
Time taken for tests:   2.368 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      3696000 bytes
HTML transferred:       3569000 bytes
Requests per second:    422.37 [#/sec] (mean)
Time per request:       2.368 [ms] (mean)
Time per request:       2.368 [ms] (mean, across all concurrent requests)
Transfer rate:          1524.49 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.0      0       0
Processing:     2    2   1.9      2      44
Waiting:        1    2   1.9      2      44
Total:          2    2   2.0      2      45

Percentage of the requests served within a certain time (ms)
  50%      2
  66%      2
  75%      2
  80%      2
  90%      2
  95%      4
  98%      6
  99%     10
 100%     45 (longest request)

## 10 concurrent requests
% ab -n 1000 -c 1 "http://127.0.0.1:8000/queryaiotest?q=test" 
Document Path:          /queryaiotest?q=test
Document Length:        3569 bytes

Concurrency Level:      1
Time taken for tests:   2.267 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      3696000 bytes
HTML transferred:       3569000 bytes
Requests per second:    441.03 [#/sec] (mean)
Time per request:       2.267 [ms] (mean)
Time per request:       2.267 [ms] (mean, across all concurrent requests)
Transfer rate:          1591.84 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.0      0       1
Processing:     2    2   1.3      2      17
Waiting:        1    2   1.2      2      16
Total:          2    2   1.3      2      17

Percentage of the requests served within a certain time (ms)
  50%      2
  66%      2
  75%      2
  80%      2
  90%      2
  95%      3
  98%      7
  99%     10
 100%     17 (longest request)

# Async DB only, with connection re-use
## 1 concurrent request
% ab -n 1000 -c 1 "http://127.0.0.1:8000/queryaiotest2?q=test"
Document Path:          /queryaiotest2?q=test
Document Length:        3569 bytes

Concurrency Level:      1
Time taken for tests:   1.531 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      3696000 bytes
HTML transferred:       3569000 bytes
Requests per second:    653.21 [#/sec] (mean)
Time per request:       1.531 [ms] (mean)
Time per request:       1.531 [ms] (mean, across all concurrent requests)
Transfer rate:          2357.67 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.0      0       0
Processing:     1    1   1.2      1      18
Waiting:        1    1   1.1      1      16
Total:          1    1   1.2      1      18

Percentage of the requests served within a certain time (ms)
  50%      1
  66%      1
  75%      1
  80%      1
  90%      2
  95%      2
  98%      5
  99%      7
 100%     18 (longest request)

## 10 concurrent requests
% ab -n 1000 -c 10 "http://127.0.0.1:8000/queryaiotest2?q=test"
Document Path:          /queryaiotest2?q=test
Document Length:        3569 bytes

Concurrency Level:      10
Time taken for tests:   1.229 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      3696000 bytes
HTML transferred:       3569000 bytes
Requests per second:    813.56 [#/sec] (mean)
Time per request:       12.292 [ms] (mean)
Time per request:       1.229 [ms] (mean, across all concurrent requests)
Transfer rate:          2936.45 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.1      0       1
Processing:     5   12   3.5     11      30
Waiting:        5   12   3.4     11      29
Total:          5   12   3.5     11      30

Percentage of the requests served within a certain time (ms)
  50%     11
  66%     12
  75%     12
  80%     13
  90%     16
  95%     21
  98%     23
  99%     29
 100%     30 (longest request)

# 100 concurrent requests
% ab -n 1000 -c 100 "http://127.0.0.1:8000/queryaiotest2?q=test"
Document Path:          /queryaiotest2?q=test
Document Length:        3569 bytes

Concurrency Level:      100
Time taken for tests:   1.233 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      3696000 bytes
HTML transferred:       3569000 bytes
Requests per second:    811.25 [#/sec] (mean)
Time per request:       123.267 [ms] (mean)
Time per request:       1.233 [ms] (mean, across all concurrent requests)
Transfer rate:          2928.09 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    1   1.3      1       5
Processing:     7  118  24.5    120     171
Waiting:        4  116  24.6    118     168
Total:          7  120  24.4    122     171

Percentage of the requests served within a certain time (ms)
  50%    122
  66%    127
  75%    136
  80%    148
  90%    152
  95%    156
  98%    161
  99%    162
 100%    171 (longest request)

# Static page only, no DB or Colbert (baseline)
## 1 concurrent request
% ab -n 1000 -c 1 "http://127.0.0.1:8000/statictest"
Document Path:          /statictest
Document Length:        28 bytes

Concurrency Level:      1
Time taken for tests:   0.901 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      153000 bytes
HTML transferred:       28000 bytes
Requests per second:    1109.38 [#/sec] (mean)
Time per request:       0.901 [ms] (mean)
Time per request:       0.901 [ms] (mean, across all concurrent requests)
Transfer rate:          165.76 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.0      0       0
Processing:     1    1   0.5      1       6
Waiting:        0    1   0.4      1       4
Total:          1    1   0.5      1       6

Percentage of the requests served within a certain time (ms)
  50%      1
  66%      1
  75%      1
  80%      1
  90%      1
  95%      2
  98%      3
  99%      3
 100%      6 (longest request)

## 10 concurrent requests
% ab -n 1000 -c 10 "http://127.0.0.1:8000/statictest"
Document Path:          /statictest
Document Length:        28 bytes

Concurrency Level:      10
Time taken for tests:   0.591 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      153000 bytes
HTML transferred:       28000 bytes
Requests per second:    1692.98 [#/sec] (mean)
Time per request:       5.907 [ms] (mean)
Time per request:       0.591 [ms] (mean, across all concurrent requests)
Transfer rate:          252.95 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.0      0       1
Processing:     2    6   2.1      5      20
Waiting:        2    6   2.1      5      20
Total:          3    6   2.1      5      20

Percentage of the requests served within a certain time (ms)
  50%      5
  66%      6
  75%      6
  80%      6
  90%      7
  95%     10
  98%     15
  99%     17
 100%     20 (longest request)