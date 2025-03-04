/*####################################################################################
$$WISEGLOVESDK, Ver 1.3 beta
$$File name		:		WISEGLOVE.h
$$Compiler		:		VC6
$$Create date	:		2012-6 Ver beta
$$Revise date	:		2012-7 Ver 1.3.2
$$Revise date	:		2014-8 Ver 1.5.0
$$History		 	:
$$Auther		 	:
$$Powered By	:		Wiseglove Tech. Co., Ltd.	<www.wiseglove.com>
####################################################################################*/
#ifndef __WISEGLOVE_H__
#define __WISEGLOVE_H__

#ifdef DLL_IMPLEMENT  
#define DLL_API __declspec(dllexport)  
#else  
#define DLL_API __declspec(dllimport)  
#endif

/*
手套标定标志，用于SetCalibMode()函数
*/
#define CALIB_AUTO  0
#define CALIB_MANU  1
typedef struct {
	float w;  //四元素
	float x;
	float y;
	float z;
} QUAT;

typedef struct {
	float x;
	float y;
	float z;
} EULER;


class DLL_API WiseGlove
{
public:

	/* ***************************************************
	*	功能：扫描系统里存在的WISEGLOVE手套
	*	参数：无
	*	返回：WISEGLOVE手套数量总数
	*/
	virtual unsigned int Scan(void) = 0;
	/* ***************************************************
	*	功能：取得串口端口号
	*	参数：num，表示在系统中的第num个手套端口
	*	返回：COM口端口号
	*/
	virtual unsigned int GetPort(unsigned int num) = 0;

	/* ***************************************************
	功能：取得手套的串口通讯波特率
	参数：num，表示在系统中的第num个手套端口
	返回：波特率数据115200或256000
	*/
	virtual unsigned int GetPortBaudrate(unsigned int num) = 0;
	/*  ***************************************************
	功能：打开WISEGLOVE端口
	参数：nPort，表示所在端口号
	返回：加载成功返回TRUE(1)， 否则返回FALSE(0)
	*/
	virtual bool  Open(unsigned int nPort, unsigned int baudrate) = 0;

	/*  ***************************************************
	功能：关闭WG数据端口
	参数：无
	返回：无
	*/
	virtual	void Close(void) = 0;

	/* ***************************************************
	*	功能：本函数只支持有手指压力传感的WISEGLOVE手套型号
	*	参数：char *filename，标定文件名字串，例如”c:\\calib.txt”
	*	返回：加载成功返回TRUE(1)， 否则返回FALSE(0)
	*/
	virtual bool LoadPressureCalib(char *filename) = 0;

	/* ***************************************************
	*	功能：在使用手动标定时，使用本函数加载标定数据文件
	*	参数：char *filename，标定文件名字串，例如”c:\\calib.txt”
	*	返回：加载成功返回TRUE(1)， 否则返回FALSE(0)
	*/
	virtual	bool LoadCalib(char *filename) = 0;

	/* ***************************************************
	*	功能：在使用手动标定的情况下，使用本函数保存标定数据
	*	参数：char *filename，标定文件名字串，例如”c:\\calib.txt”
	*	返回：加载成功返回TRUE(1)， 否则返回FALSE(0)
	*/
	virtual	bool SaveCalib(char *filename) = 0;

	/* ***************************************************
	*	功能：设置标定模式
	*	参数：int mode
	*	mode = CALIB_AUTO， 自动标定
	*	mode = 	CALIB_MANU， 手动标定
	*	返回：无
	*/
	virtual	void SetCalibMode(int mode) = 0;

	/* ***************************************************
	*	功能：传感器设置标定参数
	*	参数：int index，传感器索引值，例如19传感器型号的范围为【0，18】
	*		  unsigned short min，最小值限制
	*		  unsigned short max, 最大值限制
	*	      float angle，关节活动角度范围（度为单位）
	*	返回：如果成功 返回TRUE(1)，否则返回FALSE(0)
	*/
	virtual bool SetCalib(int index, unsigned short min, unsigned short max, float angle) = 0;

	/* ***************************************************
	*	功能：传感器设置最小值参数
	*	参数：int index，传感器索引值，范围【0，18】
	*		  unsigned short min，最小值限制
	*	返回：如果成功 返回TRUE(1)，否则返回FALSE(0)
	*/
	virtual bool SetCalibmin(int index, unsigned short min) = 0;

	/* ***************************************************
	*	功能：传感器设置最大值参数
	*	参数：int index，传感器索引值，范围【0，18】
	*		  unsigned short max，最大值限制
	*	返回：如果成功 返回TRUE(1)，否则返回FALSE(0)
	*/
	virtual bool SetCalibmax(int index, unsigned short max) = 0;

	/* ***************************************************
	*	功能：传感器设置关节活动角度范围参数
	*	参数：int index，传感器索引值，范围【0，18】
	*		  float angle，关节活动角度范围
	*	返回：如果成功 返回TRUE(1)，否则返回FALSE(0)
	*/
	virtual bool SetCalibRange(int index, float angle) = 0;

	/* ***************************************************
	*	功能：取得传感器最小值限制值
	*	参数：int index，传感器索引值，范围【0，18】
	*	返回：如果成功 返回最小值限制值，否则返回0
	*/
	virtual unsigned short GetCalibmin(int index) = 0;

	/* ***************************************************
	*	功能：取得传感器最大值限制值
	*	参数：int index，传感器索引值，范围【0，18】
	*	返回：如果成功 返回最大值限制值，否则返回0
	*/
	virtual unsigned short GetCalibmax(int index) = 0;

	/* ***************************************************
	*	功能：取得传感器关节活动角度范围
	*	参数：int index，传感器索引值，范围【0，18】
	*	返回：如果成功 返回关节活动角度范围，否则返回0
	*/
	virtual float GetCalibRange(int index) = 0;

	/* ***************************************************
	*	功能：调用本函数可以恢复所有标定值到缺省状态，标定模式恢复为自动标定模式
	*	参数：无
	*	返回：无
	*/
	virtual	void ResetCalib() = 0;

	/* ***************************************************
	*	功能：设置传感器最小角度限制
	*	参数：int index，传感器索引值，范围【0，18】
	*	      float min_adj，最小角度限制值，范围【0，0.3】
	*	返回：如果成功 返回TRUE(1)，否则返回FALSE(0)
	*/
	virtual bool SetMin_adj(int index, float min_adj) = 0;

	/* ***************************************************
	*	功能：读传感器最小角度限制
	*	参数：int index，传感器索引
	*	返回：如果成功 返回最小角度限制，否则返回0
	*/
	virtual float GetMin_adj(int index) = 0;

	/* ***************************************************
	*	功能：设置传感器最大角度限制
	*	参数：int index，传感器索引值，范围【0，18】
	*	      float max_adj，最大角度限制值，范围【0，0.3】
	*	返回：如果成功 返回TRUE(1)，否则返回FALSE(0)
	*/
	virtual bool SetMax_adj(int index, float max_adj) = 0;

	/* ***************************************************
	*	功能：读传感器最大角度限制
	*	参数：int index，传感器索引
	*	返回：如果成功 返回最大角度限制，否则返回0
	*/
	virtual float GetMax_adj(int index) = 0;

	/* ***************************************************
	*	功能：读数据手套序列号字串
	*	参数：unsigned char * sn，保存序列号字串，一般设置为16个字节
	*	返回：如果成功 返回TRUE(1)，否则返回FALSE(0)
	*/
	virtual	bool GetSn(char *sn) = 0;

	/* ***************************************************
	*	功能：读数据手套型号字串
	*	参数：unsigned char * model，保存型号字串，一般设置为16个字节
	*	返回：如果成功 返回TRUE(1)，否则返回FALSE(0)
	*/
	virtual	bool GetModel(char *model) = 0;

	/* ***************************************************
	*	功能：读数据手套厂家字串
	*	参数：unsigned char * manu，保存厂家字串，一般设置为16个字节
	*	返回：如果成功 返回TRUE(1)，否则返回FALSE(0)
	*/
	virtual	bool GetManu(char *manu) = 0;

	/* ***************************************************
	*	功能：读手指传感器数量配置
	*	参数：无
	*	返回：如果成功 返回传感器数量值，否则返回0
	*/
	virtual int  GetNumOfSensor(void) = 0;

	/* ***************************************************
	*	功能：读手指压力传感器数量
	*	参数：无
	*	返回：如果成功 返回传感器数量值，否则返回0
	*/
	virtual int  GetNumOfPressure(void) = 0;
	/* ***************************************************
	*	功能：读手臂传感器数量
	*	参数：无
	*	返回：如果成功 返回传感器数量值，否则返回0
	*/
	virtual int  GetNumOfArm(void) = 0;

	/* ***************************************************
	*	功能：读所有传感器值
	*	参数：unsigned short * data，传感器数据，最多19个ushort型数据
	*	返回：如果成功 返回当前数据包的时间戳（单位毫秒），否则返回FALSE(0)
	*/
	//virtual	int GetData(unsigned short *data)=0;
	virtual	unsigned int GetData(unsigned short *data) = 0;

	/* ***************************************************
	*	功能：读所有传感器归一化值，范围【0，4096】
	*	参数：unsigned short * data，传感器数据，最多19个ushort型数据
	*	返回：如果成功 返回当前数据包的时间戳（单位毫秒），否则返回FALSE(0)
	*/
	//virtual	int GetScaledData(unsigned short *data)=0;
	virtual	unsigned int GetScaledData(unsigned short *data) = 0;

	/* ***************************************************
	*	功能：读传感器的角度
	*	参数：float *angle，传感器角度数据，最多19个float型数据
	*	返回：如果成功 返回当前数据包的时间戳（单位毫秒），否则返回FALSE(0)
	*/
	virtual	unsigned int GetAngle(float *angle) = 0;

	/* ***************************************************
	*	功能：读手指压力传感器的原始值
	*	参数：int *pressure，传感器原始数据，最多5个int型数据
	*	返回：如果成功 返回当前数据包的时间戳（单位毫秒），否则返回FALSE(0)
	*/
	virtual unsigned int GetPressureRaw(int *pressure) = 0;

	/* ***************************************************
	*	功能：读手指压力传感器的压力
	*	参数：pressure，传感器压力数据，单位：克，最多5个int型数据
	*	返回：如果成功 返回当前数据包的时间戳（单位毫秒），否则返回FALSE(0)
	*/
	virtual unsigned int  GetPressure(int *pressure) = 0;

	/* ***************************************************
	*	功能：读手臂传感器的旋转角度
	*	参数：float *quat，3个手臂传感器的旋转角度，格式为四元素(w,x,y,z)
	*	返回： 如果成功 返回(1)，否则返回(0)
	*/
	virtual	unsigned int GetQuat(float *quat) = 0;

	/* ***************************************************
	*	功能：设置手指力反馈
	*	参数：unsigned char *fddata，5个手指力反馈数据，依次为拇指、食指、中指、无名指、小指，取值范围为32-255
	*	返回： 如果成功 返回TRUE(1)，否则返回FALSE(0)
	*/
	virtual bool SetFeedBack(unsigned char *fddata) = 0;

	/* ***************************************************
	*	功能：标定手臂传感器，本函数仅仅适合UNITY3D
	*	参数：无。注意，传感器需要安装在手臂外侧，且手臂和手掌保持自然下垂的姿势
	*	返回： 如果成功 返回TRUE(1)，否则返回FALSE(0)
	*/
	virtual void ResetQuat() = 0;

	/* ***************************************************
	*	功能：设置回调的欧拉角度，为上一次计算后的角度
	*	输入参数：float* euler，欧拉角(x,y,z)
	*	返回：无
	*/
	virtual void SetCallbackEuler(float* euler) = 0;

	/* ***************************************************
	*	功能：计算任意四元素的欧拉角度
	*	输入参数：float *quat ，四元素(w,x,y,z); 输出参数float *angle,计算后的欧拉角(x,y,z)
	*	返回：无
	*/
	virtual void GetEuler(float* quat, float *angle) = 0;

};

/* ***************************************************
 *	功能：用于新建一个手套指针，在对手套端口操作之前，首先必须要调用本函数新建一个手套指针，例如，WiseGlove* pGlove = GetWiseGlove();
 *	参数：无
 *	返回：Wiseglove型指针（备注，不管任何时候，只要调用了该函数一定要执行DelWiseGlove()销毁指针，否则会造成内存泄漏）
 */
DLL_API WiseGlove* GetWiseGlove();
/* ***************************************************
 *	功能：用于销毁一个手套指针,只要调用过GetWiseGlove()创建过手套指针，那么程序退出前，一定要执行本函数释放资源，否则会造成内存泄漏。
 *	参数：WiseGlove* p，Wiseglove型指针
 *	返回：无
 */
DLL_API void DelWiseGlove(WiseGlove* p);
//=================================================================================
#endif  //end of wiseglove.h

