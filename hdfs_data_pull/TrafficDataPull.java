/**
  *****************************************
  *****************************************
  * by Shuo Wang **
  *****************************************
  *****************************************
  */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.LineReader;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;


public class TrafficDataPull extends Configured implements Tool {
	
	public static void main ( String[] args ) throws Exception {
		
		int res = ToolRunner.run(new Configuration(), new TrafficDataPull(), args);
		System.exit(res); 
		
	} // End main
	
	public int run ( String[] args ) throws Exception {
		
		//String input = args[0];    // Input
		String temp = "Shuo/output_temp";       // Round one output
		String out = "Shuo/output";
		//String temp1 = "/scr/shuowang/lab3/exp2/temp1/";     // Round two output
		//String output1 = "/scr/shuowang/lab3/exp2/output1/";   // Round three/final output
		//String output2 = "/scr/shuowang/lab3/exp2/output2/";   // Round three/final output
		String matchtable = "Shuo/TargetSensors.csv";
		
		int reduce_tasks = 16;  // The number of reduce tasks that will be assigned to the job
		
		FileSystem fs = FileSystem.get(new Configuration());
        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(new Path(matchtable))));
        String line;
        String sensors = "";
        String orders = "";
        String directions = "";
        while ((line = br.readLine()) != null) {
        	String sensor = line.split(",")[5];
        	String order = line.split(",")[11];
        	String direction = line.split(",")[1];
        	sensors = sensors + sensor + ",";
        	orders = orders + order + ",";
        	directions = directions + direction + ",";  	
        }
        br.close();
//		String sensors = "74041,74100,87527,86935,85030,87526";
		
		Configuration conf = new Configuration();
		conf.set("sensors", sensors);
		conf.set("orders", orders);
		conf.set("directions", directions);

		Job job_one = new Job(conf, "Traffic Data Reaggregation"); 	
		job_one.setJarByClass(TrafficDataPull.class); 
		job_one.setNumReduceTasks(reduce_tasks);		
		job_one.setMapOutputKeyClass(Text.class); 
		job_one.setMapOutputValueClass(Text.class); 
		job_one.setOutputKeyClass(NullWritable.class);         
		job_one.setOutputValueClass(Text.class);
		job_one.setMapperClass(Map_One.class); 
		job_one.setReducerClass(Reduce_One.class);
		job_one.setInputFormatClass(TextInputFormat.class);  
		job_one.setOutputFormatClass(TextOutputFormat.class);
		if (args.length==0){FileInputFormat.addInputPath(job_one, new Path("Shuo/twoweekdatapull.txt"));}
		for (String input:args){FileInputFormat.addInputPath(job_one, new Path(input)); }
		FileOutputFormat.setOutputPath(job_one, new Path(temp));
		// FileOutputFormat.setOutputPath(job_one, new Path(another_output_path)); // This is not allowed
		
		// Run the job
		job_one.waitForCompletion(true); 
		
		Job job_two = new Job(conf, "Predictive Study Traffic Pull"); 	
		job_two.setJarByClass(TrafficDataPull.class); 
		job_two.setNumReduceTasks(reduce_tasks);			
		job_two.setMapOutputKeyClass(Text.class); 
		job_two.setMapOutputValueClass(Text.class); 
		job_two.setOutputKeyClass(NullWritable.class);         
		job_two.setOutputValueClass(Text.class);
		job_two.setMapperClass(Map_Two.class); 
		job_two.setReducerClass(Reduce_Two.class);
		job_two.setInputFormatClass(TextInputFormat.class);  		
		job_two.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job_two, new Path(temp));		
		// FileInputFormat.addInputPath(job_two, new Path(another_input_path)); // This is legal
		FileOutputFormat.setOutputPath(job_two, new Path(out));
		// FileOutputFormat.setOutputPath(job_two, new Path(another_output_path)); // This is not allowed
		job_two.waitForCompletion(true); 

		return 0;
	
	} // End run
	
	
	public static class Map_One extends Mapper<LongWritable, Text, Text, Text>  {		
		
		// The map method 
		public void map(LongWritable key, Text value, Context context) 
								throws IOException, InterruptedException  {
			
			// The TextInputFormat splits the data line by line.
			// So each map method receives one line (edge) from the input
			String line = value.toString();
			
			// Split the edge into two nodes 
			String[] nodes = line.split(",");
			
			if(nodes.length>=6 & (nodes.length-6)%11==0)
			{
			
			int weightedspeedsum = 0;			
			int countsum = 0;			
			int occupancysum = 0;
			double avgoccupancy = 0.0;
			double avgspeed = 0.0;
			int smallcountsum = 0;
			int middlecountsum = 0;
			int largecountsum = 0;
			
			String date = nodes[1];
			String yy = date.substring(0,4);
			String m = date.substring(4,6);
			String dd = date.substring(6,8);
			String D = m+"/"+dd+"/"+yy;
			
			String time = nodes[2];
			String hh = time.substring(0,2);
			String mm = time.substring(2,4);
			String ss = time.substring(4,6);
			int minnum = Integer.parseInt(mm)/1;			
			
			
			
			if(nodes[4].equals("failed"))
			{
				context.write(new Text(nodes[0].trim()+","+D+","+hh+","+Integer.toString(minnum)), new Text("nocomma"));				
			}
			if(nodes[4].equals("off"))
			{
				context.write(new Text(nodes[0].trim()+","+D+","+hh+","+Integer.toString(minnum)), new Text("one,comma"));				
			}
			
			
			if(nodes.length>6)
			{
				int numlanes = Integer.parseInt(nodes[5]);
				int zerospeednonzerocountflag = 0;
				
				for(int i=0;i<numlanes;i++)
				{
					if (i*11+10<=nodes.length)
					{
					String count = nodes[i*11+7];
					String speed = nodes[i*11+10];
					String occupancy = nodes[i*11+9];
					String smallcount = nodes[i*11+11];
					String middlecount = nodes[i*11+13];
					String largecount = nodes[i*11+15];
					
					if(count.equals("null"))
					{
						count = "0";
					}
					if(Integer.parseInt(count)>17)
					{
						count = "0";
					}
					if(speed.equals("null"))
					{
						speed = "0";
					}
					if(Integer.parseInt(speed)<0)
					{
						speed = "0";
					}					
					if(occupancy.equals("null"))
					{
						occupancy = "0";
					}
					if(smallcount.equals("null"))
					{
						smallcount = "0";
					}
					if(middlecount.equals("null"))
					{
						middlecount = "0";
					}
					if(largecount.equals("null"))
					{
						largecount = "0";
					}
										
					countsum += Integer.parseInt(count);							
					weightedspeedsum += Integer.parseInt(count)*Integer.parseInt(speed);					
					occupancysum += Integer.parseInt(occupancy);
					smallcountsum += Integer.parseInt(smallcount);
					middlecountsum += Integer.parseInt(middlecount);
					largecountsum += Integer.parseInt(largecount);
					
					if (Integer.parseInt(count)>0 & Integer.parseInt(speed)==0)
					{
						zerospeednonzerocountflag++;
					}
					
					}
				}
				avgoccupancy = occupancysum/numlanes;
				if (countsum>0)
				{
					avgspeed = weightedspeedsum/1.6/countsum;
				}				
				context.write(new Text(nodes[0].trim()+","+D+","+hh+","+Integer.toString(minnum)), new Text(Double.toString(avgspeed)+","+Integer.toString(countsum)+","+Double.toString(avgoccupancy)));
				if (countsum!=smallcountsum+middlecountsum+largecountsum)
				{
					context.write(new Text(nodes[0].trim()+","+D+","+hh+","+Integer.toString(minnum)), new Text("th,ree,com,ma"));
				}
				if (zerospeednonzerocountflag>0)
				{
					context.write(new Text(nodes[0].trim()+","+D+","+hh+","+Integer.toString(minnum)), new Text("fo,ur,co,mm,a"));
				}
			}					
			}
		} // End method "map"
		
	} // End Class Map_One
	
	
	// The reduce class	
	public static class Reduce_One extends Reducer<Text, Text, NullWritable, Text>  {		
		
		// The reduce method
		// For key, we have an Iterable over all values associated with this key
		// The values come in a sorted fashion.
		public void reduce(Text key, Iterable<Text> values, Context context) 
											throws IOException, InterruptedException  {
			
			int totalcount = 0;
			double totalspeed = 0.0;
			double totaloccupancy = 0.0;
			int num = 0;
			int fail = 0;
			int off = 0;
			int classmisscount = 0;
			int zerospeednonzerocount = 0;
			int missingveh = 0;
			int issue=0;
			
			for (Text val : values) {
				
				num++;
				String data = val.toString();
				
				String[] data1 = data.split(",");
				
				if (data1.length==1)
				{
					fail++;
				}
				if (data1.length==2)
				{
					off++;
				}
				if (data1.length==4)
				{
					classmisscount++;
				}
				if (data1.length==5)
				{
					zerospeednonzerocount++;
				}
				if (data1.length==3)
				{
				totalcount += Integer.parseInt(data1[1]);
				totalspeed += Double.parseDouble(data1[0])*Integer.parseInt(data1[1]);
				totaloccupancy += Double.parseDouble(data1[2]);	
				}
			}
			
			double meanspeed = 0.0;
			if(totalcount>0)
			{
				meanspeed = totalspeed/totalcount;	
			}
			double meanoccupancy = totaloccupancy/num;
			
			int hh = Integer.parseInt(key.toString().split(",")[2]);
			if (fail + off ==0 & hh>5 & hh<21 & totalcount==0)
			{
				missingveh++;
			}
			
						
			if (off>0)
			{
				issue = 10;
				context.write(NullWritable.get(),new Text(key.toString()+","+Double.toString(meanspeed)+","+Integer.toString(totalcount)+","+Double.toString(meanoccupancy)+","+Integer.toString(issue)));
			}
			if (fail>0)
			{
				issue = 20;
				context.write(NullWritable.get(),new Text(key.toString()+","+Double.toString(meanspeed)+","+Integer.toString(totalcount)+","+Double.toString(meanoccupancy)+","+Integer.toString(issue)));
			}
			if (zerospeednonzerocount>0)
			{
				issue = 30;
				context.write(NullWritable.get(),new Text(key.toString()+","+Double.toString(meanspeed)+","+Integer.toString(totalcount)+","+Double.toString(meanoccupancy)+","+Integer.toString(issue)));
			}
			if (missingveh>0)
			{
				issue = 40;
				context.write(NullWritable.get(),new Text(key.toString()+","+Double.toString(meanspeed)+","+Integer.toString(totalcount)+","+Double.toString(meanoccupancy)+","+Integer.toString(issue)));
			}
			
			
			if (classmisscount>0)
			{
				issue = 60;   
				context.write(NullWritable.get(),new Text(key.toString()+","+Double.toString(meanspeed)+","+Integer.toString(totalcount)+","+Double.toString(meanoccupancy)+","+Integer.toString(issue)));
			}
			if (fail+off+classmisscount+zerospeednonzerocount+missingveh==0)
			{
				issue = 0;
				context.write(NullWritable.get(),new Text(key.toString()+","+Double.toString(meanspeed)+","+Integer.toString(totalcount)+","+Double.toString(meanoccupancy)+","+Integer.toString(issue)));
			}
		} // End method "reduce" 
		
	} // End Class Reduce_One

	public static class Map_Two extends Mapper<LongWritable, Text, Text, Text>  {		
		public void map(LongWritable key, Text value, Context context) 
								throws IOException, InterruptedException  {
					
			Configuration conf = context.getConfiguration();
			String sensor = conf.get("sensors");
			String order = conf.get("orders");
			String direction = conf.get("directions");
			
			String[] sensors = sensor.split(",");
			String[] orders = order.split(",");
			String[] directions = direction.split(",");
			
			List<String> Sensors = Arrays.asList(sensors);
			
			String[] lines = value.toString().split(",");
			
			String date = lines[1];
			String year = date.split("/")[2];
			String month = date.split("/")[0];
			String day = date.split("/")[1];
			String hour = lines[2];
			String minute = Integer.toString(1*Integer.parseInt(lines[3]));
			
			String name = lines[0];
			if(Sensors.contains(name))
			{
				int i = Sensors.indexOf(name);
				String Order= orders[i];
				String Direction = directions[i];
				
				String speed = lines[4];
				String count = lines[5];
				String occup = lines[6];
				
				// aggregated by key
						
				context.write(new Text(year+month+day), new Text(Order+","+Direction+","+speed+","+count+","+occup+","+hour+","+minute));
				
			}
			
		} // End method "map"
		
	} // End Class Map_One

	public static class Reduce_Two extends Reducer<Text, Text, NullWritable, Text>  {		
		public void reduce(Text key, Iterable<Text> values, Context context) 
											throws IOException, InterruptedException  {
			
			Configuration conf = context.getConfiguration();
			String Order = conf.get("orders");
			String Direction = conf.get("directions");
			String[] orders = Order.split(",");
			String[] directions = Direction.split(",");
			final List<String> order_1 = new ArrayList<String>();
			final List<String> order_2 = new ArrayList<String>();
			for (int i=0;i<directions.length;i++)
			{
				if(directions[i].equals("1")){order_1.add(orders[i]);}
				if(directions[i].equals("2")){order_2.add(orders[i]);}
			}
			
			TreeMap<String,TreeMap<Integer,TreeMap<Integer,String>>> data_dir1 = new TreeMap<String,TreeMap<Integer,TreeMap<Integer,String>>>();
			TreeMap<String,TreeMap<Integer,TreeMap<Integer,String>>> data_dir2 = new TreeMap<String,TreeMap<Integer,TreeMap<Integer,String>>>();
			String[] channels = {"speed","count","occup"};
			List<String> channel = Arrays.asList(channels);
			for (String x : channel)
			{
				data_dir1.put
				(
						x,new TreeMap<Integer,TreeMap<Integer,String>>()
				);
				data_dir2.put
				(
						x,new TreeMap<Integer,TreeMap<Integer,String>>()
				);
			}
			for (String x:data_dir1.keySet())
			{
				for (String k:order_1)
				{
					data_dir1.get(x).put
					(
							Integer.parseInt(k),new TreeMap<Integer, String>()
							{
								{
								for (int h=0; h<24; h++)
					    		{
					    			for (int m5=0; m5<60; m5++)
					    			{
					    				put(h*100+m5*1, "0");
					    			}
					    		}
								}
							}
					);
				}
				for (String k:order_2)
				{
					data_dir2.get(x).put
					(
							Integer.parseInt(k),new TreeMap<Integer, String>()
							{
								{
								for (int h=0; h<24; h++)
					    		{
					    			for (int m5=0; m5<60; m5++)
					    			{
					    				put(h*100+m5*1, "0");
					    			}
					    		}
								}
							}
					);
				}
			}
			
			for (Text val : values) {
				String[] line =val.toString().split(",");
				int order = Integer.parseInt(line[0]);
				int direction = Integer.parseInt(line[1]);
				String speed = line[2];
				String count = line[3];
				String occup = line[4];
				
				int hour = Integer.parseInt(line[5]);
				int minute = Integer.parseInt(line[6]);
				int time = hour*100+minute;
				
				if (direction==1)
        		{
        			data_dir1.get("speed").get(order).put(time, speed);
        			data_dir1.get("count").get(order).put(time, count);
        			data_dir1.get("occup").get(order).put(time, occup);
           		}
				if (direction==2)
        		{
					data_dir2.get("speed").get(order).put(time, speed);
        			data_dir2.get("count").get(order).put(time, count);
        			data_dir2.get("occup").get(order).put(time, occup);
        		}
				
				//context.write(NullWritable.get(), new Text(key.toString()+","+val.toString()));

			}
			FileWriter pw1 = new FileWriter("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"_Traffic_Dir1.csv",true);
			for (String x:data_dir1.keySet())
			{
				for (int d:data_dir1.get(x).keySet())
				{
					String row = Integer.toString(channel.indexOf(x));
					for (int t:data_dir1.get(x).get(d).keySet())
					{
						row += "," + data_dir1.get(x).get(d).get(t);
					}
					pw1.append(row);
					pw1.append("\n");
				}
			}
			pw1.flush();
            pw1.close();	
			
            FileWriter pw2 = new FileWriter("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"_Traffic_Dir2.csv",true);
			for (String x:data_dir2.keySet())
			{
				for (int d:data_dir2.get(x).keySet())
				{
					String row = Integer.toString(channel.indexOf(x));
					for (int t:data_dir2.get(x).get(d).keySet())
					{
						row += "," + data_dir2.get(x).get(d).get(t);
					}
					pw2.append(row);
					pw2.append("\n");
				}
			}
			pw2.flush();
            pw2.close();	
            
            FileSystem hdfs =FileSystem.get(conf);
		    hdfs.copyFromLocalFile(new Path("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"_Traffic_Dir1.csv"),
		    		new Path("Shuo/Traffic_CSVs"));
		    //hdfs.copyFromLocalFile(new Path("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"Dir2.csv"),
		    //		new Path("Shuo/Traffic_CSVs"));
		    System.out.println("copy csv into Shuo/Traffic_CSVs");
		    
		    File f_Dir1 = new File("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"_Traffic_Dir1.csv");
		    boolean bool_TEMP = f_Dir1.delete();
		    System.out.println("File deleted: "+bool_TEMP);
		    File f_Dir2 = new File("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"_Traffic_Dir2.csv");
		    boolean bool_PREC = f_Dir2.delete();
		    System.out.println("File deleted: "+bool_PREC);
		    
		    
			
		} // End method "reduce" 
		
	} // End Class Reduce_One
 	
}
 	
 	
 	
	


