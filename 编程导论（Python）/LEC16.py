####str = input("Enter your input: "); 
####print('Received input is :', str)
##
##
##fa=open("ex0.txt","wt+")
##fo=open("ex1.txt","wb+")
##print("文件名",fo.name)
##
##list1=[1,2,3]
##fa.write(str(list1))
##fo.write(bytes(list1))
##print("是否已关闭",fo.closed)
##print("访问模式：",fo.mode)
##fa.close()
##fo.close()
##print("是否已关闭2",fo.closed)


##fo = open("foo.txt", "w") 
##fo.write("Python is a great language.\r\nYeah its great!!\r\n"); 
##fo.close() 


##fo = open("foo.txt", "r+") 
##str1 = fo.read(3); 
##print("Read String is : ", str1) 
##position = fo.tell(); 
##print("Current file position : ", position) 
##position = fo.seek(0, 0); 
##str2 = fo.read(10); 
##print("Again read String is : ", str2) 
##fo.close() 
##



fo = open("foo.txt", "r+") 
str1 = fo.read(); 
print("Read String is : ", str1) 
position = fo.tell(); 
print("Current file position : ", position) 
position = fo.seek(0, 0); 
str2 = fo.read(10); 
print("Again read String is : ", str2)
str3 = fo.readline(); 
print("third read String is : ", str3)

fo.close()
