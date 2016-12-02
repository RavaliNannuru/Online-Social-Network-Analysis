import collect
import cluster
import classify

collect.main()
cluster.main()
classify.main()

filename = "summary.txt"
file=open(filename,'w')
file.write('Number of users collected: '+ str(collect.a)+'\n')
file.write('Number of messages collected: '+ str(collect.b)+'\n')
file.write('Number of communities discovered: '+ str(cluster.c)+'\n')
file.write('Average number of users per community: '+ str(cluster.d)+'\n')   
file.write('Number of instances per class found: '+ str(classify.out)+'\n')
file.write('One example from each class: '+ str(cluster.instance1)+'\n'+str(classify.instance2)+'\n'+str(classify.instance3)+'\n')    

file.close()