using System;
using System.IO;
using System.Linq;

namespace Step1Sentiment {
    class Program {

        static readonly string[] _dataNames = { "yelp_labelled.txt", "imdb_labelled.txt", "amazon_labelled.txt" };
        static readonly string _dataPath = Path.Combine (Environment.CurrentDirectory, "Data", _dataNames[0]);
        static readonly string _modelPath = Path.Combine (Environment.CurrentDirectory, "Data", "Model.zip");
       
        static void Main (string[] args) {
            var content = File.ReadLines (_dataPath).ToList();
            Console.WriteLine(content[0]);
        }
    }
}