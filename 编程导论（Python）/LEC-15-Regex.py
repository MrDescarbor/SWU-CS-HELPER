########use find
##hand = open('mbox-short.txt')
##for line in hand:
##    line = line.rstrip()
##    if line.find('From:') >= 0:
##        print(line)

####


#######use regex
##import re 
##hand = open('mbox-short.txt')
##for line in hand:
##    line = line.rstrip()
##    if re.search('From:', line) :
##        print(line)
####
####
#######use startswith
##hand = open('mbox-short.txt')
##for line in hand:
##    line = line.rstrip()
##    if line.startswith('From:') :
##        print(line)

######
#######use regex
##import re 
##hand = open('mbox-short.txt')
##for line in hand:
##    line = line.rstrip()
##    if re.search('^From:', line) :
##        print(line)
####

##import re 
##hand = open('mbox-short.txt')
##for line in hand:
##    line = line.rstrip()
##    if re.search('^X.*:', line) :
##        print(line)


##import re 
##hand = open('mbox-short.txt')
##for line in hand:
##    line = line.rstrip()
##    if re.search('^X-\S+:', line) :
##        print(line)


####
####
##import re
##x = 'My 2 favorite numbers are 19 and 42'
##y = re.findall('[0-9]+',x)
##print(y)
####
####
####
##import re
##x = 'My 2 favorite numbers are 19 and 42'
####y = re.findall('[0-9]+',x)
####print(y) 
##y = re.findall('[AEIOU]+',x)
##print(y)
####
####
####import re
####x = 'From: Using the : character'
####y = re.findall('^F.+:', x)
####print(y)
############
############
############
############import re
##########x = 'From: Using the : character'
####y = re.findall('^F.+?:', x)
####print(y)
####
####
####
####
##import re
##x="From quzehui@swu.edu.cn Tue Nov 22  01:18:07 2022"
##y = re.findall('\S+@\S+',x)
##print(y)

##import re 
##hand = open('mbox-short.txt')
##for line in hand:
##    line = line.rstrip()
##    y = re.findall('\S+@\S+',line)
##    if len(y)>=1:
##        print(y)
##import re 
##hand = open('mbox-short.txt')
##for line in hand:
##    line = line.rstrip()
##    y = re.findall('^From (\S+@\S+)',line)
##    if len(y)>=1:
##        print(y)

##
##import re
##x="From quzehui@swu.edu.cn Tue Nov 22  01:18:07 2022"
##y = re.findall('(\S+)@(\S+)',x)
##print(y)

####
####

##data = 'From quzehui@swu.edu.cn Tue Nov 22  01:18:07 2022'
##atpos = data.find('@')
##print(atpos) 
##sppos = data.find(' ',atpos)
##print(sppos) 
##host = data[atpos+1 : sppos]
##print(host)
####
####
####
####words = line.split()
####email = words[1]
####pieces = email.split('@')
####print(pieces[1])
####
####
####
####
####import re 
####lin = 'From quzehui@swu.edu.cn Tue Nov 22  01:18:07 2022'
####y = re.findall('@([^ ]*)',lin)
####print(y)
####
####
####
##import re 
##lin = 'From quzehui@swu.edu.cn Tue Nov 22  01:18:07 2022   abc@tsinghua.edu.cn'
##y = re.findall('^From (.*?)@([^ ]+)',lin)
##print(y)
####
####
####
##import re
##hand = open('mbox-short.txt')
##numlist = list() # []
##for line in hand:
##    line = line.rstrip()
##    stuff = re.findall('^X-DSPAM-Confidence: ([0-9.]+)', line)
##    if len(stuff) != 1 :  continue
##    num = float(stuff[0])
##    numlist.append(num)
##print('Maximum:', max(numlist))

####
import re
x = 'We just received $10.00 for cookies.'
y = re.findall('\$[0-9.]+',x)
print(y)
####
