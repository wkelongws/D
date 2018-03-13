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
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.LineReader;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;




public class WeatherDataPull extends Configured implements Tool {
	
	public static void main ( String[] args ) throws Exception {
		
		int res = ToolRunner.run(new Configuration(), new WeatherDataPull(), args);
		System.exit(res); 
		
	} // End main
	
	public int run ( String[] args ) throws Exception {
		
		//String input = args[0];    // Input
		String temp = "Shuo/output_temp";       // Round one output
		//String temp1 = "/scr/shuowang/lab3/exp2/temp1/";     // Round two output
		//String output1 = "/scr/shuowang/lab3/exp2/output1/";   // Round three/final output
		//String output2 = "/scr/shuowang/lab3/exp2/output2/";   // Round three/final output
		String matchtable = "Shuo/TargetGIDs.csv";
		
		int reduce_tasks = 16;  // The number of reduce tasks that will be assigned to the job
		
		FileSystem fs = FileSystem.get(new Configuration());
        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(new Path(matchtable))));
        String line;
        String gids = "";
        String orders = "";
        String directions = "";
        while ((line = br.readLine()) != null) {
        	String gid = line.split(",")[16];
        	String order = line.split(",")[11];
        	String direction = line.split(",")[1];
        	gids = gids + gid + ",";
        	orders = orders + order + ",";
        	directions = directions + direction + ",";  	
        }
        br.close();
//		String gids = "74041,74100,87527,86935,85030,87526";
		
		Configuration conf = new Configuration();
		conf.set("gids", gids);
		conf.set("orders", orders);
		conf.set("directions", directions);

		Job job_one = new Job(conf, "PredictiveStudyWeatherPull"); 	

		job_one.setJarByClass(WeatherDataPull.class); 

		job_one.setNumReduceTasks(reduce_tasks);			
		
		job_one.setMapOutputKeyClass(Text.class); 
		job_one.setMapOutputValueClass(Text.class); 
		job_one.setOutputKeyClass(NullWritable.class);         
		job_one.setOutputValueClass(Text.class);

		job_one.setMapperClass(Map_One.class); 
		job_one.setReducerClass(Reduce_One.class);

		job_one.setInputFormatClass(TextInputFormat.class);  
		
		job_one.setOutputFormatClass(TextOutputFormat.class);

		for (String input:args){FileInputFormat.addInputPath(job_one, new Path(input)); }
		
		// FileInputFormat.addInputPath(job_one, new Path(another_input_path)); // This is legal
		FileOutputFormat.setOutputPath(job_one, new Path(temp));
		// FileOutputFormat.setOutputPath(job_one, new Path(another_output_path)); // This is not allowed

		job_one.waitForCompletion(true); 

		return 0;
	
	} // End run

	public static class Map_One extends Mapper<LongWritable, Text, Text, Text>  {		
		public void map(LongWritable key, Text value, Context context) 
								throws IOException, InterruptedException  {
					
			Configuration conf = context.getConfiguration();
			String gid = conf.get("gids");
			String order = conf.get("orders");
			String direction = conf.get("directions");
			
			String[] gids = gid.split(",");
			String[] orders = order.split(",");
			String[] directions = direction.split(",");
			
			List<String> Gids = Arrays.asList(gids);
			
			String[] lines = value.toString().split(",");
			
			String timestamp = lines[0];
			String date = timestamp.split(" ")[0];
			String time = timestamp.split(" ")[1];
			String year = date.split("-")[0];
			String month = date.split("-")[1];
			String day = date.split("-")[2];
			String hour = time.split(":")[0];
			String minute = time.split(":")[1];
			
			String id = lines[1];
			if(Gids.contains(id))
			{
				int i = Gids.indexOf(id);
				String Order= orders[i];
				String Direction = directions[i];
				
				String tmpc = lines[2];
				String wawa = lines[3];
				String ptype = lines[4];
				String dwpc = lines[5];
				String smps = lines[6];
				String drct = lines[7];
				String vsby = lines[8];
				String roadtmpc = lines[9];
				String srad = lines[10];
				String snwd = lines[11];
				String pcpn = lines[12];
				// aggregated by key
						
				context.write(new Text(year+month+day), new Text(Order+","+Direction+","+tmpc+","+wawa+","+ptype+","+dwpc
								+","+smps+","+drct+","+vsby+","+roadtmpc+","+srad+","+snwd+","+pcpn+","+hour+","+minute));
				
			}
			
		} // End method "map"
		
	} // End Class Map_One

	public static class Reduce_One extends Reducer<Text, Text, NullWritable, Text>  {		
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
			String[] channels = {"tmpc","ptype","dwpc","smps","drct","vsby","roadtmpc","srad","snwd","pcpn"};
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
					    			for (int m5=0; m5<12; m5++)
					    			{
					    				put(h*100+m5*5, "0");
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
					    			for (int m5=0; m5<12; m5++)
					    			{
					    				put(h*100+m5*5, "0");
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
				String tmpc = line[2];
				String wawa = line[3];
				String ptype_raw = line[4];
				String ptype = "0";
				if (ptype_raw.equals("3") | ptype_raw.equals("4")){ptype = "1";}
				
				String dwpc = line[5];
				String smps = line[6];
				String drct = line[7];
				String vsby = line[8];
				String roadtmpc = line[9];
				String srad = line[10];
				String snwd = line[11];
				String pcpn = line[12];
				int hour = Integer.parseInt(line[13]);
				int minute = Integer.parseInt(line[14]);
				int time = hour*100+minute;
				
				if (direction==1)
        		{
        			data_dir1.get("tmpc").get(order).put(time, tmpc);
        			//data_dir1.get("wawa").get(order).put(time, wawa);
        			data_dir1.get("ptype").get(order).put(time, ptype);
        			data_dir1.get("dwpc").get(order).put(time, dwpc);
        			data_dir1.get("smps").get(order).put(time, smps);
        			data_dir1.get("drct").get(order).put(time, drct);
        			data_dir1.get("vsby").get(order).put(time, vsby);
        			data_dir1.get("roadtmpc").get(order).put(time, roadtmpc);
        			data_dir1.get("srad").get(order).put(time, srad);
        			data_dir1.get("snwd").get(order).put(time, snwd);
        			data_dir1.get("pcpn").get(order).put(time, pcpn);
        		}
				if (direction==2)
        		{
        			data_dir2.get("tmpc").get(order).put(time, tmpc);
        			//data_dir2.get("wawa").get(order).put(time, wawa);
        			data_dir2.get("ptype").get(order).put(time, ptype);
        			data_dir2.get("dwpc").get(order).put(time, dwpc);
        			data_dir2.get("smps").get(order).put(time, smps);
        			data_dir2.get("drct").get(order).put(time, drct);
        			data_dir2.get("vsby").get(order).put(time, vsby);
        			data_dir2.get("roadtmpc").get(order).put(time, roadtmpc);
        			data_dir2.get("srad").get(order).put(time, srad);
        			data_dir2.get("snwd").get(order).put(time, snwd);
        			data_dir2.get("pcpn").get(order).put(time, pcpn);
        		}
				
				//context.write(NullWritable.get(), new Text(key.toString()+","+val.toString()));

			}
			FileWriter pw1 = new FileWriter("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"Dir1.csv",true);
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
			
            FileWriter pw2 = new FileWriter("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"Dir2.csv",true);
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
		    hdfs.copyFromLocalFile(new Path("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"Dir1.csv"),
		    		new Path("Shuo/CSVs"));
		    //hdfs.copyFromLocalFile(new Path("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"Dir2.csv"),
		    //		new Path("Shuo/CSVs"));
		    System.out.println("copy csv into Shuo/CSVs");
		    
		    File f_Dir1 = new File("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"Dir1.csv");
		    boolean bool_TEMP = f_Dir1.delete();
		    System.out.println("File deleted: "+bool_TEMP);
		    File f_Dir2 = new File("/hadoop/yarn/local/usercache/team/appcache/"+key.toString()+"Dir2.csv");
		    boolean bool_PREC = f_Dir2.delete();
		    System.out.println("File deleted: "+bool_PREC);
		    
		    
			
		} // End method "reduce" 
		
	} // End Class Reduce_One
 	
}
 	
 	
 	
	


