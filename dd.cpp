#include"stdio.h"
#include"string.h"
int a[1000000];
int main()
{
	int m,n,j,i,max,flag=0;
	memset(a,0,sizeof(a));
	while(scanf("%d %d",&n,&m))
	{
		if(m==-1&&n==-1)
			break;
		if(max<m)
			max=m;
		a[m]=n;
	}
	for(i=max;i>=1;i--)
	{
		if(a[i]!=0)
		{
			flag=1;
			j=i;
			printf("%d %d ",a[i]*i,j-1);
		}
	}
	if(flag==0)
		printf("0");
	return 0;
}
